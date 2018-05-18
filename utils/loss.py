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
import cifar10
import cifar10_input
import modules
from candidate import Candidate

def _learning_rate_decay_fn(learning_rate, global_step):
  return tf.train.exponential_decay(
      learning_rate,
      global_step,
      decay_steps=1000,
      decay_rate=0.9,
      staircase=True)

def get_acc_and_train_op(FLAGS, var_list_to_learn, output, tr_label, input_graph):
  MOVING_AVERAGE_DECAY = 0.997
  with input_graph.as_default():
    # global step
    _init = tf.constant(0)
    global_step =  tf.Variable(_init,name = 'global_step', trainable=False)
    # Cross Entropy
    with tf.name_scope('cross_entropy'):
      diff = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tr_label, logits=output)
      with tf.name_scope('total'):
        # L2 Regularize
        regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/50000)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, weights_list = var_list_to_learn)
        loss = tf.reduce_mean(diff) + reg_term
    
    # Accuracy 
    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.nn.in_top_k(output, tr_label, 1)
        #correct_prediction = tf.equal(tf.argmax(y, 1), y_)#tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # loss_avg
    opt = tf.contrib.layers.optimize_loss(loss, global_step, FLAGS.learning_rate, 'Adam',
                                          gradient_noise_scale=None, gradient_multipliers=None,
                                          clip_gradients=None, #moving_average_decay=0.9,
                                          learning_rate_decay_fn=_learning_rate_decay_fn, update_ops=None, variables=var_list_to_learn, name=None)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step, name='average')
    ema_op = ema.apply([loss, accuracy] + var_list_to_learn)#tf.trainable_variables())
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)
    
    loss_avg = ema.average(loss)
    tf.summary.scalar('loss/training', loss_avg)
    accuracy_avg = ema.average(accuracy)
    tf.summary.scalar('accuracy/training', accuracy_avg)
    check_loss = tf.check_numerics(loss, 'model diverged: loss->nan')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([opt]):
      train_op = tf.group(*updates_collection)

    return train_op, accuracy_avg, global_step, ema
