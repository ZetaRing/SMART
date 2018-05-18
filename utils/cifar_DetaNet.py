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

def generate_scale(input_graph, scale_name):
  with input_graph.as_default():
    _init = tf.truncated_normal(shape=[1], mean=1, stddev=0.1)
    with tf.name_scope(scale_name + "_scale"):
      scale = tf.Variable(_init)
  return scale
  
def build_module(moduel_type, input_data, filter_num, keep_prob, module_name, input_graph):
  with input_graph.as_default():
    out = None
    w = None
    b = None

    in_scale = generate_scale(input_graph, module_name + "_in")
    input_data = in_scale * input_data

    if moduel_type == "conv":
      out, w, b = modules.conv_module(
            input_tensor = input_data,
            filt_num = filter_num,
            kernel_size = [3,3],
            is_active = 1,
            stride = 1,
            layer_name = module_name,
            keep_prob = keep_prob
      )
    elif moduel_type == "fire":
      out, w, b = modules.fire_layer(
        input_tensor = input_data,
        filter_num = filter_num,
        is_active = 1,
        layer_name = module_name,
        keep_prob = keep_prob,
      )
    elif moduel_type == "DR":
      out, w, b = modules.Dimensionality_reduction_module(
        input_tensor =input_data,
        is_active = 1,
        layer_name = module_name
      )     
    else:
      raise ValueError("Unknow module type")
    # print('the inscale is:'+str()in_scale)
  return out, w, b, [in_scale]


def build_previous_layer(moduel_type, input_data, module_number, filter_num, keep_prob, layer_name, input_graph):
  with input_graph.as_default():
    out = []
    w = []
    b = []
    scale = []
    for i in range(module_number):
      module_name = layer_name + "_" + str(i)
      _out , _w, _b, _scale = build_module(moduel_type, input_data, filter_num, keep_prob, module_name, input_graph )
      out.append(_out)
      w += _w
      b += _b
      scale += _scale
    # print('attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
    # print('the preinscale is:'+str(scale))  
  return out, w, b, scale


def build_extend_layer(moduel_type, input_data, previous_module_number, module_number, filter_num, keep_prob, layer_name, input_graph):
  with input_graph.as_default():
    out = []
    w = []
    b = []
    scale = []
    for i in range(module_number):
      module_name = layer_name + "_" + str(i + previous_module_number)
      _out , _w, _b, _scale = build_module(moduel_type, input_data, filter_num, keep_prob, module_name, input_graph )
      out.append(_out)
      w += _w
      b += _b
      scale += _scale
  return out, w, b, scale


def build_layer(moduel_type, input_data, pre_module_number, ext_module_number, filter_num, keep_prob, layer_name, input_graph):
  with input_graph.as_default():
    var_list_to_learn = []
    var_list_to_restore = []
    pre_out_scale = [generate_scale(input_graph, layer_name + "_" + str(i) + "_out") for i in range(pre_module_number)]
    ext_out_scale = [generate_scale(input_graph, layer_name + "_" + str(i) + "_out") for i in range(pre_module_number, ext_module_number + pre_module_number)]
    #for i in range(ext_module_number):
    #  ext_out_scale.append( [generate_scale(input_graph, layer_name + "_ext")] + [generate_scale(input_graph, "ext_add_pre") for j in range(pre_module_number)])
    out = None

    pre_out, pre_w, pre_b, pre_in_scale = build_previous_layer(moduel_type, input_data, pre_module_number, filter_num, keep_prob, layer_name, input_graph)
    ext_out, ext_w, ext_b, ext_in_scale = build_extend_layer(moduel_type, input_data, pre_module_number, ext_module_number, filter_num, keep_prob, layer_name, input_graph)

    out = sum(map(lambda a,b: a*b, pre_out + ext_out, pre_out_scale + ext_out_scale)) / (pre_module_number + ext_module_number)
    # if moduel_type == "fire" or moduel_type == "conv":
    #   if ext_module_number != 0:
    #     _concated_var = []
    #     for i in ext_out:
    #       _var_to_concat = []

    #       _tmp = generate_scale(input_graph, layer_name + "_ext")
    #       _var_to_concat.append(_tmp * i)
    #       ext_out_scale.append(_tmp)

    #       for j in pre_out:
    #         _tmp = generate_scale(input_graph, layer_name + "ext_add_pre")
    #         _var_to_concat.append(_tmp * j)
    #         ext_out_scale.append(_tmp)
          
    #       _concated_var.append(tf.concat(_var_to_concat, 3))
    #     _ext_sum = sum(_concated_var)/ext_module_number

    #   if pre_module_number != 0:
    #     _pre_sum = sum(map(lambda a,b: a*b, pre_out, pre_out_scale))/pre_module_number

    #   if ext_module_number and pre_module_number:
    #     out = tf.concat([_pre_sum, _ext_sum], 3)
    #   elif ext_module_number:
    #     out = _ext_sum
    #   elif pre_module_number:
    #     out = _pre_sum
    #   else:
    #     raise ValueError("Module width is 0")

    # elif moduel_type == "DR":
    #   ext_out_scale = [generate_scale(input_graph, layer_name + "_ext") for i in range(ext_module_number)]
    #   out = sum(map(lambda a,b: a*b, pre_out + ext_out, pre_out_scale + ext_out_scale))/ (pre_module_number + ext_module_number)
    
    # # because fire module's parameter's will change according to input channel
    # # due to our transfer strategy, the input channel will change when transfer
    # # so, can not save or reload the parameters in fire module
    # if moduel_type == "fire":
    #   var_list_to_learn = ext_w + ext_b + pre_in_scale + ext_in_scale + pre_out_scale + ext_out_scale + pre_w + pre_b
    #   var_list_to_restore = []
    #   var_list_to_save = ext_w  + var_list_to_restore # + ext_b
    # else:
    #   var_list_to_learn = ext_w + ext_b + pre_in_scale + ext_in_scale + pre_out_scale + ext_out_scale
    #   var_list_to_restore = pre_w #+ pre_b
    #   var_list_to_save = ext_w + var_list_to_restore # + ext_b
    # print(pre_out_scale)
    # print(pre_in_scale)
    var_list_to_learn = ext_w + ext_b + ext_in_scale + ext_out_scale + pre_in_scale + pre_out_scale
    var_list_to_restore = pre_w + pre_b + pre_in_scale + pre_out_scale
    var_list_to_save = ext_w + ext_b + ext_in_scale + ext_out_scale + pre_in_scale + pre_out_scale +  pre_w + pre_b

    # print(moduel_type, end = " ")
    # print(pre_module_number, end = " ")
    # print(ext_module_number, end = " ")
    # print(filter_num, end = " ")
    # print(out)
  # print(var_list_to_save)
  # print(pre_out_scale)
  return out, var_list_to_restore, var_list_to_learn, var_list_to_save


def build_graph(FLAGS, pre_structure, ext_structure, input_data, image_size, output_num, keep_prob, input_graph):
  #pre_structure.display_structure()
  #ext_structure.display_structure()
  with input_graph.as_default():
    #------------------------------------------------
    # define Local Variables
    #------------------------------------------------
    pre_FR = pre_structure.feature_layer_num
    pre_FC = pre_structure.fc_layer_num

    pre_FA = pre_structure.feature_layer_array
    pre_MA = pre_structure.module_num_array

    pre_F = pre_structure.filter_num * 2  # due to filter number must be an even number


    ext_FR = ext_structure.feature_layer_num
    ext_FC = ext_structure.fc_layer_num

    ext_FA = ext_structure.feature_layer_array
    ext_MA = ext_structure.module_num_array

    # result
    var_list_to_learn = []
    var_list_to_restore = []
    var_list_to_save = []
    out = None

    #------------------------------------------------
    # define Input
    #------------------------------------------------
    # Input
    with tf.name_scope('input'):
      x = input_data

    #------------------------------------------------
    # define Graph
    #------------------------------------------------

    # first conv layer
    i = 0
    _type = "conv"
    out, _restore, _learn, _save = build_layer(
      moduel_type = _type,
      input_data = x,
      pre_module_number = pre_MA[i],
      ext_module_number = ext_MA[i],
      filter_num = pre_F,
      keep_prob = keep_prob,
      layer_name = "conv_layer" + str(i),
      input_graph = input_graph
      )
    var_list_to_restore += _restore
    var_list_to_learn += _learn
    var_list_to_save += _save

    # feature abstract layer
    _length = min(pre_FR, ext_FR)

    # print("------------------debug-----------------")
    # pre_structure.display_structure()
    # ext_structure.display_structure()
    # print("----------------------------------------")

    for i in range(_length):
      # type
      if pre_FA[i] == 1:
        _type = "DR"
      elif pre_FA[i] == 0:
        _type = "fire"
      else:
        raise ValueError("Unknow module encode")

      # build
      out, _restore, _learn, _save = build_layer(
        moduel_type = _type,
        input_data = out,
        pre_module_number = pre_MA[i + 1],
        ext_module_number = ext_MA[i + 1],
        filter_num = pre_F,
        keep_prob = keep_prob,
        layer_name = "feature_layer_" + str(i + 1),
        input_graph = input_graph
        )
      var_list_to_restore += _restore
      var_list_to_learn += _learn
      var_list_to_save += _save
    
      # dangerious area: unless you understand and clear, don't change
    if( pre_FR > ext_FR):
      _abort_out = out
      for i in range(ext_FR, pre_FR):
        # type
        if pre_FA[i] == 1:
          _type = "DR"
        elif pre_FA[i] == 0:
          _type = "fire"
        else:
          raise ValueError("Unknow module encode")

        # build
        _abort_out, _restore, _learn, _save = build_layer(
          moduel_type = _type,
          input_data = _abort_out,
          pre_module_number = pre_MA[i + 1],
          ext_module_number = 0,
          filter_num = pre_F,
          keep_prob = keep_prob,
          layer_name = "feature_layer_" + str(i + 1),
          input_graph = input_graph
          )
        var_list_to_restore += _restore
        var_list_to_save += _save
    
    if( pre_FR < ext_FR):
      for i in range(pre_FR, ext_FR):
        # type
        if ext_FA[i] == 1:
          _type = "DR"
        elif ext_FA[i] == 0:
          _type = "fire"
        else:
          raise ValueError("Unknow module encode")

        # build
        out, _restore, _learn, _save = build_layer(
          moduel_type = _type,
          input_data = out,
          pre_module_number = 0,
          ext_module_number = ext_MA[i + 1],
          filter_num = pre_F,
          keep_prob = keep_prob,
          layer_name = "feature_layer_" + str(i + 1),
          input_graph = input_graph
          )
        var_list_to_restore += _restore
        var_list_to_learn += _learn
        var_list_to_save += _save
    
    #print(out)
    # full connection layer
      # reshape
    _shape = out.shape[1:]
    _length = 1
    for _i in _shape:
        _length *= int(_i)
    out=tf.reshape(out, [-1,_length])
    #print(out)

    # full connection
    for i in range(ext_FC):
      tmp_out = []
      _M = ext_MA[i + ext_FR + 1]
      for j in range(_M):
        _out, _weights, _biases = modules.fc_layer(
          input_tensor = out, 
          filt_num = pre_F,
          layer_name = "fc_layer_" + str(i) + "_" + str(j),
          keep_prob = keep_prob
        )
        tmp_out.append(_out) 
        var_list_to_learn += _weights
        var_list_to_learn += _biases
      out = sum(tmp_out)/_M
    #print(out)

    # output layer
    y, output_weights ,output_biases = modules.nn_layer(out, output_num, 'output_layer')
    var_list_to_learn += output_weights
    var_list_to_learn += output_biases
    # print(y)
  return y, var_list_to_restore, var_list_to_learn, var_list_to_save

if __name__ == "__main__":
  sess = tf.InteractiveSession()
  a = generate_scale(tf.get_default_graph(), "a")
  print(a)
