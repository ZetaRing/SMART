from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import scipy.io as sio
import tensorflow.python.platform
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24
TFRECORD_FILE = "svhn_train.tfrecords"

def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  # check whether generated tfrecords
  if not os.path.exists(os.path.join(data_dir, TFRECORD_FILE)):
    generate_tfrecord(data_dir)
    print("Successfully generate svhn tfrecords!")  

  # read tfrecord
  filenames = [os.path.join(data_dir, TFRECORD_FILE) ]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([os.path.join(data_dir, TFRECORD_FILE)])
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
  images = tf.decode_raw(features['image_raw'], tf.float32)
  images = tf.reshape(images, [32, 32, 3])
  labels = tf.cast(features['label'], tf.int32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(images, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])


  num_preprocess_threads = 1
  min_queue_examples = 50
  images_batch, label_batch = tf.train.shuffle_batch(
      [float_image, labels],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  print ('Filling queue with svhn images before starting to train. '
         'This will take a few minutes.')


  return images_batch, label_batch

# create tfrecord files
def get_svhn_list(data_dir):
  # Get SVHN dataset
  file_name=os.path.join(data_dir,"train_32x32.mat");
  train=sio.loadmat(file_name);
  tr_data_svhn=np.zeros((len(train['y']),32*32*3),dtype=np.float32);
  tr_label_svhn=np.zeros((len(train['y']),10),dtype=np.int32);
  for i in range(len(train['y'])):
    tr_data_svhn[i]=np.reshape(train['X'][:,:,:,i],[1,32*32*3]);
    tr_label_svhn[i,train['y'][i][0]-1]=1.0;
  tr_data_svhn=tr_data_svhn/255.0;

  data_len_svhn=len(tr_label_svhn)
  return tr_data_svhn, tr_label_svhn, data_len_svhn

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_tfrecord(data_dir):
    images, labels, num_examples = get_svhn_list(data_dir)
    #print(np.array(images).shape)
    #print(num_examples)
    filename = os.path.join(data_dir,TFRECORD_FILE)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples - 1):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
  train_data, train_label = distorted_inputs("/dataset/svhn/", 32)