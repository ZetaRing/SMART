import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

TFRECORD_FILE = "./tmp/binary_mnist.tfrecords"
IMAGE_SIZE = 28
CHANNEL_NUM = 1

def read_train_data(FLAGS, num1, num2):
    if not os.path.exists(TFRECORD_FILE):
        generate_tfrecord(FLAGS, num1, num2)
        print("Successfully generate binary_mnist tfrecords!")
    tr_data, tr_label = read_tfrecords(FLAGS)
    return tr_data, tr_label


def get_mnist(FLAGS, num1, num2):
    mnist = read_data_sets("/dataset/mnist/", dtype=tf.float32, one_hot=True)
    total_tr_data, total_tr_label = mnist.train.next_batch(mnist.train._num_examples)
    
    # Gathering a1 Data
    tr_data_a1=total_tr_data[(total_tr_label[:,num1]==1.0)]
    # add noise
    for i in range(len(tr_data_a1)):
        for j in range(len(tr_data_a1[0])):
            rand_num=np.random.rand()
            if(rand_num>=0.5):
                tr_data_a1[i,j]=np.minimum(tr_data_a1[i,j]+rand_num,1.0)
    
    # Gathering a2 Data
    tr_data_a2=total_tr_data[(total_tr_label[:,num2]==1.0)]
    for i in range(len(tr_data_a2)):
        for j in range(len(tr_data_a2[0])):
            rand_num=np.random.rand()
            if(rand_num>=0.5):
                tr_data_a2[i,j]=np.minimum(tr_data_a2[i,j]+rand_num,1.0)
        
    tr_data1=np.append(tr_data_a1,tr_data_a2,axis=0)
    tr_label1=np.zeros((len(tr_data1),2),dtype=float)
    for i in range(len(tr_data1)):
        if(i<len(tr_data_a1)):
            tr_label1[i,0]=1.0
        else:
            tr_label1[i,1]=1.0
    print(tr_data1)
    print(tr_label1)
    return tr_data1, tr_label1, len(tr_label1)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_tfrecord(FLAGS, num1, num2):
    images, labels, num_examples = get_mnist(FLAGS, num1, num2)
    print(np.array(images).shape)
    print(num_examples)
    filename = TFRECORD_FILE
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples - 1):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_tfrecords(FLAGS):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([TFRECORD_FILE])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    images = tf.decode_raw(features['image_raw'], tf.float32)
    images = tf.reshape(images, [28, 28, 1])
    #images = tf.cast(images, tf.float32)
    labels = tf.cast(features['label'], tf.int32)

    num_preprocess_threads = 1
    min_queue_examples = 50
    images_batch, label_batch = tf.train.shuffle_batch(
        [images, labels],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples)

    return images_batch, label_batch







