from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys,os,time
import subprocess
import scipy.io as sio
import tensorflow as tf
from six.moves import urllib
import numpy as np

sys.path.append("utils/")
import modules
from candidate import Candidate
import cifar_DetaNet
import loss
import dataset

def train(FLAGS, prefix_structure, candidate, data_set, max_steps):
  with tf.Graph().as_default():
    # # check prefix_structure
    # if(prefix_structure == None):
    #     _prefix_structure = candidate.create_empty_prefix()
    # else:
    #     _prefix_structure = prefix_structure

    # dataset
    tr_data, tr_label, image_size, channel_num, output_num = dataset.get_data(FLAGS, data_set)
    tr_data = tf.reshape(tr_data, [-1, image_size, image_size, channel_num])

    # build graph
    with tf.name_scope('keep_prob'):
      keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    out, _restore, _learn, _save = cifar_DetaNet.build_graph(
      FLAGS = FLAGS,
      pre_structure = prefix_structure, 
      ext_structure = candidate, 
      input_data = tr_data,
      image_size = image_size, 
      output_num = output_num,
      keep_prob = keep_prob, 
      input_graph = tf.get_default_graph()
    )

    train_op, accuracy_avg, global_step, ema = loss.get_acc_and_train_op(
      FLAGS = FLAGS,
      var_list_to_learn = _learn,
      output = out,
      tr_label = tr_label,
      input_graph = tf.get_default_graph()
    )

    # init
      # first init all variables than restore
    sess = tf.InteractiveSession()
    #saver = tf.train.Saver(tf.global_variables())
    # print(tf.global_variables())
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    if(FLAGS.use_moving_avg):
      # need to change..................................................
      variables_to_restore = ema.variables_to_restore()  # restore shadow variables to use moving average variables
    else:
      variables_to_restore = _restore
    
    # restore
    if variables_to_restore:
      saver1 = tf.train.Saver(variables_to_restore)
      # Restores from checkpoint
      ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver1.restore(sess, ckpt.model_checkpoint_path)

    # save
    if _save:
      # print(_save)
      saver2 = tf.train.Saver(_save)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  # Learning & Evaluating
    init_acc = 0
    final_acc = 0
    total_acc = 0
    for k in range(max_steps):
      _, acc_geo_tmp = sess.run([train_op, accuracy_avg], feed_dict={keep_prob:FLAGS.dropout})
      total_acc += acc_geo_tmp
      #print(acc_geo_tmp)
      if(k%100 == 0 and k > 0):
        if FLAGS.use_moving_avg:
          print("step %d, moving_avg_acc %f" %(k, acc_geo_tmp))
        else:
          print("step %d, current_acc %f" %(k, acc_geo_tmp))

      if(k == 100):init_acc = acc_geo_tmp
      if(k == max_steps - 1):
        final_acc = acc_geo_tmp

      
      # only after evolution, we can override saver file
      if( k == max_steps -1 and max_steps > FLAGS.T):
        print("saving...", end="  ")
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        saver2.save(sess, checkpoint_path)
        print("done.")

      #if(k == max_steps - 1):
        # print("---------------scales----------------------")
        # _scales = []
        # for i in _learn:
        #   if "scale" in i.name:
        #     _scales.append(i)
        # _scales_value = sess.run(_scales)
        # for i in zip(_scales, _scales_value):
        #   print(i[0].name + ": \t" + str(i[1][0]))
  


    coord.request_stop()  
    coord.join(threads)
    sess.close()

    return final_acc - init_acc, final_acc #total_acc / max_steps