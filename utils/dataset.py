import numpy as np

import cifar10
import cifar10_input
import cifar100_input
import binary_mnist_input
import svhn


def get_data(FLAGS, dataset):
    tr_data = None
    tr_label = None
    image_size = None
    channel_num = None
    output_num = None
    if dataset == 'cifar10':
        tr_data, tr_label = cifar10_input.distorted_inputs(FLAGS.cifar_data_dir, FLAGS.batch_size)
        image_size = cifar10_input.IMAGE_SIZE
        channel_num = 3
        output_num = 10
    elif dataset == 'svhn':
        tr_data, tr_label = svhn.distorted_inputs(FLAGS.svhn_data_dir, FLAGS.batch_size)
        image_size = svhn.IMAGE_SIZE
        channel_num = 3
        output_num = 10
    elif dataset == 'cifar20':
        tr_data, tr_label = cifar100_input.distorted_inputs(20, FLAGS.cifar100_data_dir, FLAGS.batch_size)
        image_size = cifar100_input.IMAGE_SIZE
        channel_num = 3
        output_num = 20
    elif dataset == 'mnist1':
        tr_data, tr_label = binary_mnist_input.read_train_data(FLAGS, FLAGS.a1, FLAGS.a2)
        image_size = 28
        channel_num = 1
        output_num = 2
    elif dataset == 'mnist2':
        tr_data, tr_label = binary_mnist_input.read_train_data(FLAGS, FLAGS.b1, FLAGS.b2)
        image_size = 28
        channel_num = 1
        output_num = 2
    elif dataset == 'mnist3':
        tr_data, tr_label = binary_mnist_input.read_train_data(FLAGS, FLAGS.c1, FLAGS.c2)
        image_size = 28
        channel_num = 1
        output_num = 2
    elif dataset == 'mnist4':
        tr_data, tr_label = binary_mnist_input.read_train_data(FLAGS, FLAGS.d1, FLAGS.d2)
        image_size = 28
        channel_num = 1
        output_num = 2
    else:
        raise ValueError("No such dataset")

    return tr_data, tr_label, image_size, channel_num, output_num

  