from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time

import cifar10
import cifar10_input
import modules
from candidate import Candidate
import cifar_DetaNet
import train

def evolution_algo(FLAGS, data_set, prefix_structure):
    '''
    return structure is the extend part
    not include the prefix_structure
    '''
    # if prefix_structure != None:
    #     prefix_structure.display_structure()

    # ceate inti candidates
    candidates = [Candidate(prefix_structure) for i in range(FLAGS.candi)] 

    # evolution algo
    _best1 = 0
    _best2 = 0
    _worst1 = 0
    _worst2 = 0
    best_index = 0
    for step in range(FLAGS.max_generations):
        # train and evaluate
        _start_time = time.time()
        acc = []
        slope = []
        for i in candidates:
            _slope ,_acc = train.train(
                FLAGS = FLAGS,
                prefix_structure = prefix_structure,
                candidate = i,
                data_set = data_set,
                max_steps = FLAGS.T
            )
            acc += [_acc]
            slope += [_slope]

        # find best and worst
        _best1 = slope.index(sorted(slope)[-1])
        _best2 = slope.index(sorted(slope)[-2])
        _worst1 = slope.index(sorted(slope)[0])
        _worst2 = slope.index(sorted(slope)[1])
        _worst3 = slope.index(sorted(slope)[2])
        _worst4 = slope.index(sorted(slope)[3])

        best_index = _best1

        # create offsprings
        _offspring1 = candidates[_best1].copy()
        _offspring2 = candidates[_best1].copy()
        _offspring3 = candidates[_best2].copy()
        _offspring4 = candidates[_best2].copy()

        _offspring1.mutation()
        _offspring2.mutation()
        _offspring3.mutation()
        _offspring4.mutation()


        # survivor selection
        candidates[_worst1] = _offspring1
        candidates[_worst2] = _offspring2
        candidates[_worst3] = _offspring3
        candidates[_worst4] = _offspring4

        _time_cost = time.time() - _start_time
        print("-----------------evolution results---------------- ")
        print(acc)
        print("best extend structure: ")
        candidates[_best1].display_structure()
        print("generation: %d, training_avg_acc: %f, time_cost: %f s" % (step, max(acc), _time_cost))
        print("")

    return candidates[best_index]



