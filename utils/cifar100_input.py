# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sys
from six.moves import urllib
import tarfile

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.

# NUM_CLASSES = 10
#IMAGE_SIZE = 32
IMAGE_DEPTH = 3
NUM_CLASSES_CIFAR10 = 10
NUM_CLASSES_CIFAR20 = 20
NUM_CLASSES_CIFAR100 = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

print('...cifar_input...')

CIFAR10_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
CIFAR100_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'

def maybe_download_and_extract(data_dir,data_url=CIFAR10_DATA_URL):
    dest_directory = data_dir #'../CIFAR10_dataset'
    DATA_URL = data_url
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1] #'cifar-10-binary.tar.gz'
    filepath = os.path.join(dest_directory, filename)#'../CIFAR10_dataset\\cifar-10-binary.tar.gz'
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    if data_url== CIFAR10_DATA_URL:
        extracted_dir_path = os.path.join(dest_directory,'cifar-10-batches-bin')  # '../CIFAR10_dataset\\cifar-10-batches-bin'
    else :
        extracted_dir_path = os.path.join(dest_directory, 'cifar-100-binary')  # '../CIFAR10_dataset\\cifar-10-batches-bin'
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def read_cifar10(filename_queue,coarse_or_fine=None):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.

  #<1 x label><3072 x pixel>
  #...
  #<1 x label><3072 x pixel>

  label_bytes = 1  # 2 for CIFAR-100

  result.height = 32
  result.width = 32
  result.depth = 3

  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes,header_bytes=0,footer_bytes=0)

  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def read_cifar100(filename_queue,coarse_or_fine='fine'):
  """Reads and parses examples from CIFAR100 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  class CIFAR100Record(object):
    pass
  result = CIFAR100Record()


  coarse_label_bytes = 1
  fine_label_bytes = 1
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth

  record_bytes = coarse_label_bytes + fine_label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes,header_bytes=0,footer_bytes=0)
  result.key, value = reader.read(filename_queue)

  record_bytes = tf.decode_raw(value, tf.uint8)

  coarse_label = tf.cast(tf.strided_slice(record_bytes, [0], [coarse_label_bytes]), tf.int32)

  fine_label = tf.cast(tf.strided_slice(record_bytes, [coarse_label_bytes], [coarse_label_bytes + fine_label_bytes]), tf.int32)

  if coarse_or_fine == 'fine':
    result.label = fine_label #
  else:
    result.label = coarse_label #

  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [coarse_label_bytes + fine_label_bytes],
                       [coarse_label_bytes + fine_label_bytes + image_bytes]),
                        [result.depth, result.height, result.width])

  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(cifar10or20or100 ,data_dir, batch_size):
  """
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  if cifar10or20or100 == 10:
    filenames = [os.path.join(data_dir,'data_batch_%d.bin' % i) for i in xrange(1,6)]
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    coarse_or_fine = None
  if cifar10or20or100 == 20:
    filenames = [os.path.join(data_dir,'train.bin')]
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar100(filename_queue, 'coarse')
  if cifar10or20or100 == 100:
    filenames = [os.path.join(data_dir, 'train.bin')]
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar100(filename_queue, 'fine')

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  casted_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  padded_image = tf.image.resize_image_with_crop_or_pad(casted_image,width+4,height+4)

  distorted_image = tf.random_crop(padded_image, [height, width, 3])

  distorted_image = tf.image.random_flip_left_right(distorted_image)

  distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)

  float_image = tf.image.per_image_standardization(distorted_image)

  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR%d images before starting to train. '
         'This will take a few minutes.' % (min_queue_examples, cifar10or20or100))

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def inputs(cifar10or20or100, eval_data, data_dir, batch_size):
  """
  return:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  if cifar10or20or100 == 10:
      read_cifar = read_cifar10
      coarse_or_fine = None
      if not eval_data:
          filenames = [os.path.join(data_dir,'data_batch_%d.bin' % i) for i in xrange(1,6)]
          num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
      else:
          filenames = [os.path.join(data_dir,'test_batch.bin')]
          num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  if cifar10or20or100 == 20 or cifar10or20or100 == 100:
      read_cifar = read_cifar100
      if not eval_data:
          filenames = [os.path.join(data_dir,'train.bin')]
          num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
      else:
          filenames = [os.path.join(data_dir,'test.bin')]
          num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  if cifar10or20or100 == 100:
      coarse_or_fine = 'fine'
  if cifar10or20or100 == 20:
      coarse_or_fine = 'coarse'

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_cifar(filename_queue, coarse_or_fine = coarse_or_fine)
  casted_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(casted_image,width,height)

  float_image = tf.image.per_image_standardization(resized_image)

  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

# APIS
# read in list form
# training set and evaluate set
def read_list_data(FLAGS):
  # Get CIFAR 10  dataset
  cifar10.maybe_download_and_extract()
  tr_label_cifar10=np.zeros((50000,10),dtype=float)
  ts_label_cifar10=np.zeros((10000,10),dtype=float)

  for i in range(1,6):
    file_name=os.path.join(FLAGS.cifar_data_dir,"data_batch_"+str(i)+".bin")
    f = open(file_name,"rb")
    data=np.reshape(bytearray(f.read()),[10000,3073])
    if(i==1):
      tr_data_cifar10=data[:,1:]/255.0
    else:
      tr_data_cifar10=np.append(tr_data_cifar10,data[:,1:]/255.0,axis=0)
    for j in range(len(data)):
      tr_label_cifar10[(i-1)*10000+j,data[j,0]]=1.0

  file_name=os.path.join(FLAGS.cifar_data_dir,"test_batch.bin")
  f = open(file_name,"rb")
  data=np.reshape(bytearray(f.read()),[10000,3073])
  print(data.shape)
  for i in range(len(data)):
    ts_label_cifar10[i,data[i,0]]=1.0
  ts_data_cifar10=data[:,1:]/255.0
  print(np.shape(ts_data_cifar10))

  data_num_len_cifar10=len(tr_label_cifar10)
  ts_num_len_cifar10=len(ts_label_cifar10)
  
  return tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10, ts_data_cifar10, ts_label_cifar10, ts_num_len_cifar10

# read in tensor form, need start queue
# only training set
def read_train_data(FLAGS):
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.cifar_data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = distorted_inputs(data_dir=FLAGS.cifar_data_dir,
                                    batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

# read in tensor form, need start queue
# only test set
def read_test_data(FLAGS):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.cifar_data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = inputs(eval_data=True,
                          data_dir=FLAGS.cifar_data_dir,
                          batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)

  return images, labels


if __name__ == "__main__":
  tr_data, tr_label = distorted_inputs(20, "/dataset/cifar100/cifar-100-binary/", 32)