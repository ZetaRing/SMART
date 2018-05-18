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
import shutil  

sys.path.append("utils/")
import cifar10
import cifar10_input
import modules
from candidate import Candidate
import cifar_DetaNet
import evolution_algo
import train

FLAGS = None

def main(_):
  # clear save files
  shutil.rmtree(FLAGS.save_dir)  
  os.mkdir(FLAGS.save_dir)
  shutil.rmtree("./tmp/")  
  os.mkdir("./tmp/")
  #os.remove("/dataset/svhn/svhn_train.tfrecords")
  
  seed_network = Candidate()
  seed_network.set_empty_prefix()
  structure1 = task(data_set = 'cifar20', prefix_structure = seed_network)
  structure2 = task(data_set = 'cifar10', prefix_structure = structure1)
  structure3 = task(data_set = 'svhn', prefix_structure = structure2)


def task(data_set, prefix_structure = None):
  # evolution
  best_candidate = evolution_algo.evolution_algo(
    FLAGS = FLAGS,
    data_set = data_set,
    prefix_structure = prefix_structure
  )
  # combine
  if prefix_structure != None:
    best_structure = prefix_structure.extend(best_candidate)
  else:
    best_structure = best_candidate
  # train
  best_structure.display_structure()
  '''
  train structure should be desperate to restore previous fixed weights
  '''
  _, final_avg_acc = train.train(
    FLAGS = FLAGS,
    prefix_structure = prefix_structure, 
    candidate = best_candidate, 
    data_set = data_set, 
    max_steps = FLAGS.max_step
  )

  # display
  print("------------------transfer results------------------")
  print("best structure training_avg_acc "+ str(final_avg_acc))
  best_structure.display_structure()
  compurtation_of_network = best_structure.compurtation_of_network()
  print("compurtation_of_network :  " + str( )+ str(compurtation_of_network)) 
  print("")

  return best_structure


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  
  parser.add_argument('--save_dir', type=str, default='./save/transfer/',
                      help='save trained argues')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Initial learning rate')
  parser.add_argument('--T', type=int, default=3000,
                      help='The Number of steps per geopath')
  parser.add_argument('--batch_size', type=int, default=256,
                      help='The Number of batches per each geopath')
  parser.add_argument('--candi', type=int, default=6,
                      help='The Number of Candidates of geopath, should greater than 4')
  parser.add_argument('--max_generations', type = int,default = 6,
                      help='The Generation Number of Evolution')
  parser.add_argument('--max_step', type = int,default = 15000,#200000,
                      help='The max training step of final structure')
  parser.add_argument('--dropout', type=float, default=1,
                      help='Keep probability for training dropout.')
  parser.add_argument('--use_fp16', type=bool, default=False,
                      help='Use float16 in netwrok or not.')
  parser.add_argument('--use_moving_avg', type=bool, default=False,#True,
                      help='Use shadow variables to replace origal variables for evaluation or not when restorage variables')
  # cifar
  parser.add_argument('--cifar_data_dir', type=str, default='/dataset/cifar_data/cifar-10-batches-bin/',
                      help='Directory for storing input data')
  parser.add_argument('--cifar100_data_dir', type=str, default='/dataset/cifar100/cifar-100-binary/',
                      help='Directory for storing input data')
  # svhn
  parser.add_argument('--svhn_data_dir', type=str, default='/dataset/svhn/',
                      help='Directory for storing input data')
  # mnist argus
  parser.add_argument('--data_dir', type=str, default='/dataset/mnist/',
                      help='Directory for storing input data')
  parser.add_argument('--a1', type=int, default=1,
                      help='The first class of task1')
  parser.add_argument('--a2', type=int, default=3,
                      help='The second class of task1')
  parser.add_argument('--b1', type=int, default=1,
                      help='The first class of task2')
  parser.add_argument('--b2', type=int, default=2,
                      help='The second class of task2')
  parser.add_argument('--c1', type=int, default=3,
                      help='The first class of task3')
  parser.add_argument('--c2', type=int, default=4,
                      help='The second class of task3')
  parser.add_argument('--d1', type=int, default=4,
                      help='The first class of task4')
  parser.add_argument('--d2', type=int, default=5,
                      help='The second class of task4')
              

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)