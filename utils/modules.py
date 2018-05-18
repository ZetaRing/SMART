from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np

def parameters_backup(var_list_to_learn):
  var_list_backup=np.zeros(len(var_list_to_learn),dtype=object);
  for i in range(len(var_list_to_learn)):
    var_list_backup[i]=var_list_to_learn[i].eval();
  return var_list_backup;

def parameters_update(sess,var_update_placeholders,var_update_ops,var_list_backup):
  for i in range(len(var_update_placeholders)):
    sess.run(var_update_ops[i],{var_update_placeholders[i]:var_list_backup[i]});
    
def geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,candi,L,M):
  for i in range(L):
    for j in range(M):
      sess.run(geopath_update_ops[i,j],{geopath_update_placeholders[i,j]:candi[i,j]});

def geopath_initializer(L,M):
  geopath=np.zeros((L,M),dtype=object);
  for i in range(L):
    for j in range(M):
      geopath[i,j]=tf.Variable(1.0);
  return geopath;

def mutation(geopath,L,M,N):
  for i in range(L):
    for j in range(M):
      if(geopath[i,j]==1):
        rand_value=int(np.random.rand()*L*N);
        if(rand_value<=1):
          geopath[i,j]=0;
          rand_value2=np.random.randint(-2,2);
          rand_value2=rand_value2-2;
          if(((j+rand_value2)>=0)&((j+rand_value2)<M)):
            geopath[i,j+rand_value2]=1;
          else:
            if((j+rand_value2)<0):
              geopath[i,0]=1;
            elif((j+rand_value2)>=M):
              geopath[i,M-1]=1;
  return geopath;

def select_two_candi(M):
  selected=np.zeros(2,dtype=int);
  j=0;
  while j<=2:
    rand_value=int(np.random.rand()*M);
    if(j==0):
      selected[j]=rand_value;j+=1;
    else:
      if(selected[0]!=rand_value):
        selected[j]=rand_value;j+=1;
        break;
  return selected[0],selected[1];
  
def get_geopath(L,M,N):
  geopath=np.zeros((L,M),dtype=float);
  for i in range(L):
    j=0;
    #Active module # can be smaller than N
    while j<N:
      rand_value=int(np.random.rand()*M);
      if(geopath[i,rand_value]==0.0):
        geopath[i,rand_value]=1.0;j+=1;
  return geopath;
      

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def module_weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  kl = shape[0]
  dl = shape[-1]
  stddev = math.sqrt(2. /(kl**2 * dl))
  initial = tf.truncated_normal(shape, stddev=stddev)
  return [tf.Variable(initial)];

def module_weight_variable_fc(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return [tf.Variable(initial)];
  

def module_bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return [tf.Variable(initial)];

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def batch_norm(x):
  epsilon = 1e-3
  batch_mean, batch_var = tf.nn.moments(x, [0])
  return tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, epsilon)

# full connection layer
def fc_layer(input_tensor, filt_num, layer_name, keep_prob , act=tf.nn.relu):
  # init
  weights=module_weight_variable_fc([int(input_tensor.shape[-1]), filt_num])
  biases=module_bias_variable([filt_num])
  #print([int(input_tensor.shape[-1]), filt_num])

  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights[0]) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    activations = tf.nn.dropout(activations, keep_prob)
    tf.summary.histogram('activations', activations)
    return activations, weights, biases

# hidden layer module, include three kinds of modules in a layer
def module2(i,input_tensor, filt_num, is_active, layer_name, act=tf.nn.relu):
  # init
  weights = module_weight_variable([int(input_tensor.shape[-1]), filt_num])
  biases = module_bias_variable([filt_num])
  
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # Skip Layer
    if(i%3==0):
      return input_tensor * is_active, weights, biases
    # Linear Layer with relu
    elif(i%3==1):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        variable_summaries(weights[0])
      with tf.name_scope('biases'):
        variable_summaries(biases[0])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights[0]) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations * is_active, weights, biases
    # Residual Layer with relu
    elif(i%3==2):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        variable_summaries(weights[0])
      with tf.name_scope('biases'):
        variable_summaries(biases[0])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights[0]) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')+input_tensor
      tf.summary.histogram('activations', activations)
      return activations * is_active, weights, biases

def conv_module(input_tensor, filt_num, kernel_size, is_active, stride, layer_name, keep_prob ,act=tf.nn.relu):
  '''conv layer
  Args:
    input_tensor: output of former layer or input training data.
                  should be a map in the shape of square. reshape should be done before input
    filt_num:     number of filters in this module
    kerner_size:  filter size in conv, format [kernel_height, kernel_width]
    is_active:    whether is module is actived
    stride:       stride in conv
    layer_name:   name of layer
    act:          active function
  '''
  # init
  conv_kernel = module_weight_variable(list(kernel_size)+[int(input_tensor.shape[-1]),filt_num])
  biases=module_bias_variable([filt_num])
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('conv_kernel'):
      variable_summaries(conv_kernel[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('conv_plus_b'):
      preactivate = tf.nn.conv2d(input_tensor, filter=conv_kernel[0],strides=[1,stride,stride,1],padding="VALID") + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = batch_norm(preactivate)
    activations = act(activations, name='activation')
    
    #activations = tf.nn.dropout(activations, keep_prob)

    tf.summary.histogram('activations', activations)
    return activations * is_active, conv_kernel, biases

def nn_layer(input_tensor, result_num, layer_name, act=tf.nn.relu):
  ''' Output layer
  '''
  #init
  weights = module_weight_variable([int(input_tensor.shape[-1]),result_num])
  biases = module_bias_variable([result_num])
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights[0]) + biases
      tf.summary.histogram('pre_activations', preactivate)
    return preactivate, weights, biases

def _variable_with_weight_decay(name, wd, initial_value, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, initial_value, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _variable_on_device(name, initial_value, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    initial_value: initial value and shape for variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  var = tf.Variable(name = name, initial_value=initial_value, trainable=trainable)
  return var

def _conv_layer(layer_name, input_tensor, keep_prob, filters, size, stride, padding,freeze=False, relu=True, stddev=0.001):
  with tf.variable_scope(layer_name) as scope:
    channels = input_tensor.get_shape()[3]

    #kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)=
    #bias_init = tf.constant_initializer(0.0)
    kernel_init = tf.random_normal(shape=[size, size, int(channels), filters], stddev=stddev)
    bias_init = tf.zeros(shape=[filters])

    kernel = _variable_with_weight_decay('kernels', wd=0.0001 , initial_value=kernel_init, trainable=(not freeze))
    biases = _variable_on_device('biases',initial_value=bias_init, trainable=(not freeze))

    conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding=padding, name='convolution')
    conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
    out = batch_norm(conv_bias)
    if relu:
      out = tf.nn.relu(out, 'relu')
    else:
      out = out
    
    out = tf.nn.dropout(out, keep_prob)
    return out, kernel, biases


def _conv_layer_without(layer_name, input_tensor, filters, size, stride, padding,freeze=False, relu=True, stddev=0.001):
  with tf.variable_scope(layer_name) as scope:
    channels = input_tensor.get_shape()[3]

    #kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)=
    #bias_init = tf.constant_initializer(0.0)
    kernel_init = tf.random_normal(shape=[size, size, int(channels), filters], stddev=stddev)
    bias_init = tf.zeros(shape=[filters])

    kernel = _variable_with_weight_decay('kernels', wd=0.0001 , initial_value=kernel_init, trainable=(not freeze))
    biases = _variable_on_device('biases',initial_value=bias_init, trainable=(not freeze))

    conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding=padding, name='convolution')
    conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
    out = batch_norm(conv_bias)
    if relu:
      out = tf.nn.relu(out, 'relu')
    else:
      out = out
    
    return out, kernel, biases


  
def _max_pooling_layer(input_tensor, kernel_size, stride, padding, layer_name):
  """Max pooling layer operation constructor.
  Args:
    layer_name: layer name.
    input_tensor: input tensor
    size: kernel size.
    stride: stride
    padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    Returns:  A pooling layer operation.
              last two return values is for unify interface with other modules.
  
    """
  with tf.variable_scope(layer_name) as scope:
    out =  tf.nn.max_pool(input_tensor, ksize=[1, kernel_size, kernel_size, 1], 
                          strides=[1, stride, stride, 1],padding=padding)
    activation_size = np.prod(out.get_shape().as_list()[1:])
    return out


def fire_layer(input_tensor, filter_num, is_active, layer_name, keep_prob, stddev=0.01,freeze=False):
  """Fire layer constructor.

  Args:
    layer_name: layer name
    input_tensor: input tensor
    s1x1: number of 1x1 filters in squeeze layer.
    e1x1: number of 1x1 filters in expand layer.
    e3x3: number of 3x3 filters in expand layer.
    freeze: if true, do not train parameters in this layer.
  Returns:
    fire layer operation.
  """
  kernels = []
  biases = []
  s1x1 = filter_num
  s1x1_1 = int(s1x1 / 3)
  e1x1 = int(s1x1 / 2)
  e3x3 = s1x1 - e1x1

  sq1x1, _kernel, _bias = _conv_layer(layer_name+'/squeeze1x1', input_tensor, keep_prob , filters=s1x1_1, size=1, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  ex1x1 ,_kernel, _bias= _conv_layer(layer_name+'/expand1x1', sq1x1, keep_prob ,filters=e1x1, size=1, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  ex3x3 ,_kernel, _bias= _conv_layer(layer_name+'/expand3x3', sq1x1, keep_prob, filters=e3x3, size=3, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  concat_result = tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
  
  return concat_result* is_active, kernels, biases




def res_fire_layer(input_tensor, is_active, layer_name, keep_prob, layer_num, total_layer, stddev=0.01,freeze=False):
  """Fire layer constructor.

  Args:
    layer_name: layer name
    input_tensor: input tensor
    s1x1: number of 1x1 filters in squeeze layer.
    e1x1: number of 1x1 filters in expand layer.
    e3x3: number of 3x3 filters in expand layer.
    freeze: if true, do not train parameters in this layer.
  Returns:
    fire layer operation.
  """
  kernels = []
  biases = []
  s1x1 = int(input_tensor.get_shape()[3])
  s1x1_1 = int(s1x1 / 3)
  e1x1 = int(s1x1 / 2)
  e3x3 = s1x1 - e1x1

  sq1x1, _kernel, _bias = _conv_layer(layer_name+'/squeeze1x1', input_tensor, keep_prob , filters=s1x1_1, size=1, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  ex1x1 ,_kernel, _bias= _conv_layer(layer_name+'/expand1x1', sq1x1,keep_prob ,filters=e1x1, size=1, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  ex3x3 ,_kernel, _bias= _conv_layer(layer_name+'/expand3x3', sq1x1, keep_prob, filters=e3x3, size=3, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  concat_result = tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

  _decay_rate = 1.0 / (total_layer - layer_num)
  
  return (concat_result + input_tensor * _decay_rate )* is_active, kernels, biases


def res_module(input_tensor, is_active, layer_name, stddev=0.01, freeze=False):
  """res layer constructor.
  Args:
    layer_name: layer name
    input_tensor: input tensor
    channels: number of 3x3 filters in first_layer of res_module.
    channels: number of 3x3 filters in second_layer of res_module.
    always we think f3x3 should be equal to s3x3, but there might be some differences, isn't it?
    freeze: if true, do not train parameters in this layer.
  Returns:
    res_module operation.
  """
  kernels = []
  biases = []

  channels = int(input_tensor.get_shape()[3])
  feature_map_of_firstlayer, _kernel, _bias = _conv_layer(layer_name+'/first_layer', input_tensor, filters=channels, size= 3, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  feature_map_of_secondlayer, _kernel, _bias = _conv_layer(layer_name+'/second_layer', feature_map_of_firstlayer, filters=channels, size=3, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  kernels.append(_kernel)
  biases.append(_bias)

  return (input_tensor + feature_map_of_secondlayer) * is_active, kernels, biases


def Dimensionality_reduction_module(input_tensor, is_active, layer_name, stddev=0.01,freeze=False):
  """Dimensionality_reduction layer constructor.
  Args:
    layer_name: layer name
    input_tensor: input tensor
    c3x3: number of 3x3 filters convolution
    freeze: if true, do not train parameters in this layer.
  Returns:
    Dimensionality_reduction operation.
  """
  c3x3 = int(int(input_tensor.get_shape()[3]) / 2)
  feature_map_of_pooling = _max_pooling_layer(input_tensor,  kernel_size= 2, stride=2, padding='VALID', layer_name = layer_name+'/pooling')
  feature_map_of_convolution, _kernel, _bias = _conv_layer_without(layer_name+'/convolution', input_tensor, filters=c3x3, size=2, stride=2,
    padding='VALID', stddev=stddev, freeze=freeze)
  return  tf.concat([feature_map_of_pooling, feature_map_of_convolution], 3, name=layer_name+'/concat_pc') * is_active, [_kernel], [_bias]

def conv(input_tensor, filt_num, kernel_size, layer_name, keep_prob, stride=1,  act=tf.nn.relu):
  # init
  conv_kernel = module_weight_variable(list(kernel_size)+[int(input_tensor.shape[-1]),filt_num])
  biases=module_bias_variable([filt_num])
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('conv_kernel'):
      variable_summaries(conv_kernel[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('conv_plus_b'):
      preactivate = tf.nn.conv2d(input_tensor, filter=conv_kernel[0],strides=[1,stride,stride,1],padding="SAME") + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = batch_norm(preactivate)
    activations = act(activations, name='activation')
    activations = tf.nn.dropout(activations,keep_prob)
    tf.summary.histogram('activations', activations)
    return activations, conv_kernel, biases

# error
def conv_pooling(input_tensor, filt_num, conv_kernel_size, pooling_kernel_size, layer_name, conv_stride, pooling_stride, act=tf.nn.relu):
  # init
  conv_kernel = module_weight_variable(list(conv_kernel_size)+[int(input_tensor.shape[-1]),filt_num])
  biases=module_bias_variable([filt_num])
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('conv_plus_b'):
      preactivate = tf.nn.conv2d(input_tensor, filter=conv_kernel[0],strides=[1,conv_stride,conv_stride,1],padding="SAME") + biases
    with tf.name_scope('max_pooling'):
      out =  tf.nn.max_pool(input_tensor, ksize=[1, pooling_kernel_size, pooling_kernel_size, 1], 
                          strides=[1, pooling_stride, pooling_stride, 1],padding="SAME")
    activations = batch_norm(preactivate)
    activations = act(activations, name='activation')
    
    return activations , conv_kernel, biases

