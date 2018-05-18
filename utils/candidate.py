import numpy as np
import copy
import random

class Candidate:
    ''' A data struct to restore candidate struct info during evoluation algo. '''
    # evolution argus
        # mutation argus
    mutation_rate = 0.2
        # range argus
    # Attention !!!!!!!!!!!!!
    # the max value of any values should satisify max * mutation rate >= 1 or mutation() will not work!!!!
    minFr = 2
    maxFr = 10
    minFc = 0
    maxFc = 2

    # set at __init__()
    maxDR = 3           # Max Dimension Reduction layer allowed, to avoid reduce below 1x1.
                        # Should set according to input and set below maxFr
    startCheckLayer = 0 # Due to transfer, DR check is limit to last layers

    minM = 1
    maxM = 20
    minFl = 6 / 2       # due to filter number must be an even number for fire and demonsion layer
    maxFl = 20 / 2      # useless, set at 'create_empty_prefix'


    # struct level 1
    feature_layer_num = 0  # feature abstract layers numbers
    feature_layer_array = [1 for i in range(feature_layer_num)] # 0 means conv layer, 1 means pool layer
    disable_mask = [0 for i in range(maxFr)]        # disable input is 1*1's Dimensionality_reduction_module
    fc_layer_num = 1       # full connection layers numbers
    out_feature_layer = 0
    # struct level 2
    module_num_array = [1 for i in range(feature_layer_num + fc_layer_num + 1)]  # module number in each layer
                                                                                 # +1 for first conv layer
    # struct level 3
    filter_num = 1          # filters in each module, must be an even number
    
    #input_shape = 32
    input_channel = 3

    # random generate candidates
    # def __init__(self, maxDR = 3, startCheckLayer = 0):
    #     self.feature_layer_num = self._get_random_num(self.minFr, self.maxFr)
    #     self.out_feature_layer = self.feature_layer_num

    #     # limit Dimension Reduction layer number
    #     while(True):
    #         self.feature_layer_array = [np.random.randint(2) for i in range(self.feature_layer_num)] # 0 or 1 sequence
    #         if(sum(self.feature_layer_array[self.startCheckLayer:]) <= self.maxDR):break

    #     self.fc_layer_num = self._get_random_num(self.minFc, self.maxFc)

    #     self.filter_num = self._get_random_num(self.minFl, self.maxFl)

    #     self.module_num_array = [self._get_random_num(self.minM, self.maxM) for i in range(self.fc_layer_num + self.feature_layer_num + 1)]

    #     self.maxDR = maxDR
    #     self.startCheckLayer = startCheckLayer


    def __init__(self, prefix_structure = None):
        
        if(prefix_structure == None): return  # prefix_structure == None means this is a seed network

        self.feature_layer_num = prefix_structure.feature_layer_num
        self.out_feature_layer = prefix_structure.feature_layer_num

        self.feature_layer_array = prefix_structure.feature_layer_array

        self.fc_layer_num = 1

        self.filter_num = prefix_structure.filter_num

        self.module_num_array =  [0 for i in range(self.feature_layer_num+1)] + [20]

        self.startCheckLayer = len(prefix_structure.feature_layer_array)
        self.maxDR = prefix_structure.maxDR - sum(prefix_structure.feature_layer_array)
        

    # first task need emtpy prefix but have the same filter number with the ext_structure
    # seed network
    def set_empty_prefix(self):
        self.feature_layer_num = 1
        self.out_feature_layer = 1
        self.feature_layer_array = [1]
        self.fc_layer_num = 1       
        self.module_num_array = [1, 1, 1]   # for first conv layer
        self.filter_num = 20/2        # filter_num is set fixed through whole process!!!

    def copy(self):
        return copy.deepcopy(self)

    def extend(self, ext_structure):
        '''
        must use pre_structure extend to ext_structure
        '''
        res = Candidate()
        # feature layer
        res.feature_layer_num = max(self.feature_layer_num, ext_structure.feature_layer_num)
        res.out_feature_layer = ext_structure.out_feature_layer

        if self.feature_layer_num >= ext_structure.feature_layer_num:
            res.feature_layer_array = self.feature_layer_array
        else:
            res.feature_layer_array = self.feature_layer_array
            res.feature_layer_array += ext_structure.feature_layer_array[self.feature_layer_num:]
        
        # fc layer
        res.fc_layer_num = ext_structure.fc_layer_num

        # module number
        res.module_num_array = []
            # fr part and conv part
        _ext_length = ext_structure.feature_layer_num + 1   # +1 for conv layer
        _pre_length = self.feature_layer_num + 1
        min_length = min( _ext_length, _pre_length)
        max_length = max( _ext_length, _pre_length)
        for i in range(min_length):
            res.module_num_array.append( self.module_num_array[i] + ext_structure.module_num_array[i] )
        if _ext_length > _pre_length:
            res.module_num_array += ext_structure.module_num_array[min_length : max_length]
        elif _ext_length < _pre_length:
            res.module_num_array += self.module_num_array[min_length : max_length]
            # fc part
        res.module_num_array += ext_structure.module_num_array[ 1 + ext_structure.feature_layer_num :]

        # filter number
        res.filter_num = self.filter_num
        
        return res


    def compurtation_of_network(self):
        compurtation_of_fl = 28* 28* self.filter_num* self.module_num_array[0]* self.input_channel
        i = 0
        output_size = 28
        while i < self.out_feature_layer:
            if self.feature_layer_array[i] == 0:
                compurtation = self.module_num_array[i + 1]* (output_size* output_size* 1* 1 * self.filter_num* self.filter_num + output_size *output_size*self.filter_num* self.filter_num/2 *100) 
                compurtation_of_fl = compurtation_of_fl +compurtation
            else :  
                output_size = np.floor(output_size/2)
                compurtation = output_size* output_size* self.filter_num * self.filter_num/2 *3 *3
                compurtation_of_fl = compurtation_of_fl + compurtation
            i = i + 1
        return compurtation_of_fl
   
    def _get_random_num(self, min_value, max_value, init=True ):
        '''
        # this random generator is based on that 
        # min_value is near 0
        # +-3*sigma value represent 99% possible
        _gap = max_value - min_value
        _value = np.random.normal(0,_gap/3.0,1)
        _value = int(abs(_value)) + min_value
        if _value > max_value:
            _value = max_value
        '''
        if(init == True):
            _value = np.random.rand() * (max_value - min_value + 1) + min_value 
        else:
            _value = np.random.rand() * (max_value - min_value) + min_value 
        
        return int(_value)

    def mutation(self): # apply mutation to this candidate
        # feature layer number and feature layer array
        while(True):
            _feature_layer_array = copy.deepcopy(self.feature_layer_array)
            _module_num_array = copy.deepcopy(self.module_num_array[1: self.feature_layer_num + 1])
            _feature_layer_num = int((1 - self.mutation_rate) * self.feature_layer_num + self.mutation_rate * self._get_random_num(self.minFr, self.maxFr))
            
            _dst = _feature_layer_num - len(_feature_layer_array)
            if(_dst < 0):
                for i in range(abs(_dst)):
                    _index = np.random.randint(len(_feature_layer_array))
                    del _feature_layer_array[_index]
                    del _module_num_array[_index]

            elif(_dst > 0):
                _feature_layer_array += [np.random.randint(2) for i in range(_dst)]
                _module_num_array += [self._get_random_num(self.minM, self.maxM) for i in range(_dst)]

            for i in range(int(self.mutation_rate * _feature_layer_num)):
                _index = np.random.randint(_feature_layer_num)
                _feature_layer_array[_index] = not _feature_layer_array[_index]
            
            if(sum(_feature_layer_array[self.startCheckLayer:]) <= self.maxDR):
                self.module_num_array = [self.module_num_array[0]] + _module_num_array + self.module_num_array[1 + self.feature_layer_num :]
                self.feature_layer_num = _feature_layer_num
                self.feature_layer_array = _feature_layer_array
                break
        self.out_feature_layer = self.feature_layer_num

        # fc and filter
        #_fc_layer_num = int((1 - self.mutation_rate) * self.fc_layer_num + self.mutation_rate * self._get_random_num(self.minFc, self.maxFc))
        _fc_layer_num = -1
        while (_fc_layer_num > self.maxFc or _fc_layer_num < self.minFc):
            _fc_layer_num = int(random.uniform( (self.maxFc + self.maxFr)/2 + 0.5, (self.maxFc - self.minFc) / 2))

        _dst = _fc_layer_num - self.fc_layer_num
        if _dst > 0:
            self.module_num_array += [self._get_random_num(self.minM, self.maxM) for i in range(_dst)]
        elif _dst < 0:
            self.module_num_array = self.module_num_array[:_dst]
        self.fc_layer_num = _fc_layer_num

        self.filter_num = int((1 - self.mutation_rate) * self.filter_num + self.mutation_rate * self._get_random_num(self.minFl, self.maxFl))

        # module number array
        for i in range(len(self.module_num_array)):
            self.module_num_array[i] = int((1-self.mutation_rate) * self.module_num_array[i] + 
                                            self.mutation_rate * self._get_random_num(self.minM, self.maxM))
            if self.module_num_array[i] == 0 : self.module_num_array[i] = 1


    # discard
    def crossover(self, parentA, parentB): # inheir genotype from parents
        _min = int(min(parentA.feature_layer_num ,parentB.feature_layer_num))
        cross_point = np.random.randint(_min)
        self.feature_layer_num = parentB.feature_layer_num
        self.feature_layer_array = parentA.feature_layer_array[:cross_point] + parentB.feature_layer_array[cross_point:]

        _min = min(parentA.fc_layer_num, parentB.fc_layer_num)
        _max = max(parentA.fc_layer_num, parentB.fc_layer_num)
        self.fc_layer_num = self._get_random_num(_min, _max, False)

        _min = min(parentA.module_num, parentB.module_num)
        _max = max(parentA.module_num, parentB.module_num)
        self.fc_layer_num = self._get_random_num(_min, _max, False)

        _min = min(parentA.filter_num, parentB.filter_num)
        _max = max(parentA.filter_num, parentB.filter_num)
        self.fc_layer_num = self._get_random_num(_min, _max, False)

    def display_structure(self): # show network struct info represented by this candidate
        if(sum(self.disable_mask) != 0):
            print("some layer is hidden due to feature map to small")
        #print("feature layer number: ")
        #print(self.feature_layer_num)
        print("feature array, 0 - conv and 1 - pooling: ")
        print([self.feature_layer_array[i] for i in range(int(self.feature_layer_num)) if self.disable_mask[i] != 1])
        print("Used feature layer number: ")
        print(self.out_feature_layer)
        print("full connection layer number: ")
        print(int(self.fc_layer_num))
        print("module number in each layer: ")
        print(self.module_num_array)
        print("filter number in each module: only first evolution result is vaild !!!!!!")
        print(int(self.filter_num)  * 2)
        print("")


if __name__ == "__main__":
    a = Candidate()
    b = Candidate()
    c = a.extend(b)
    a.display_structure()
    b.display_structure()
    c.display_structure()
