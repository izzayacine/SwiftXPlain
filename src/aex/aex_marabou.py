#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Marabou Oracle
#
################################################################################
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from timeit import default_timer as timer
import onnx
import numpy as np

from .aex_oracle import AExOracle

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent/'Marabou'))
from maraboupy import Marabou, MarabouCore
################################################################################


class Oracle_marabou(AExOracle):
    def __init__(self, nn_filename):
        super().__init__()

        # load the neural network
        if nn_filename.endswith('.onnx'):
            self.neural_network = Marabou.read_onnx(nn_filename)
        elif nn_filename.endswith('.nnet'):
            self.neural_network = Marabou.read_nnet(nn_filename)
        else:
            raise Exception('Unrecognized file format')

    def delete(self):
        """
            Delete the neural network and the robustness tool.
        """
        self.neural_network.clear()
        del self.neural_network
        self.neural_network = None

    def encode_aex(self, instance, epsilon):
        """
            Parse the input instance and prepare constraints for finding adversarial examples.
        """
        
        self.in_values, self.out_label = instance
        self.epsilon = epsilon
        # flatten the input and output variables
        self.flatten_inputVars = self.neural_network.inputVars[0].flatten()
        self.flatten_outputVars = self.neural_network.outputVars[0].flatten()
        
        sample = np.array(instance[0]) # use original shape not flatten
        if len(sample.shape) == 3:
            sample = np.expand_dims(sample, axis=0) # (1, channels, h, w)
            
        #assert (len(sample.shape) == 4)
        #self.image_type = 'grayscale' if (sample.shape[1] == 1) else 'rgb' # gray: 1 channel, rgb: 3 channels

        net_out = np.argmax(self.neural_network.evaluateWithoutMarabou(sample))
        assert (self.out_label == net_out), f'target={self.out_label}, marabou output={net_out}'
        # clear constraint and bounds
        self.neural_network.clearProperty()
        self.neural_network.additionalEquList.clear()
        self.neural_network.disjunctionList.clear()
    
    def has_aex(self, fixed=[], timeout=60, verbose=0) -> bool:
        """
            Check if there exist adversarial examples.
        """
        # options = Marabou.createOptions(numWorkers=4,
        #                                 timeoutInSeconds=timeout,
        #                                 snc=True,
        #                                 solveWithMILP=True,
        #                                 milpTightening="lp",
        #                                 milpSolverTimeout=5,
        #                                 lpSolver="gurobi",
        #                                 verbosity=0)
        options = Marabou.createOptions(timeoutInSeconds=timeout, verbosity=0)

        # input_shape = self.neural_network.inputVars[0].shape
        input_shape = self.in_values.shape
        # Create two copies of self.in_values
        input_lb = np.copy(self.in_values)
        input_ub = np.copy(self.in_values)
        
        assert (len(input_shape) == 3)
        
        n_pixels = input_shape[-1]*input_shape[-2]
        hypos = np.full((n_pixels,), False, dtype=bool)
        for i in fixed:
            hypos[i] = True         
        
        # assume the last two dimensions are height and width
        for i in range(n_pixels):
            if not hypos[i]:
                # Update lower and upper bounds based on fixed indices
                # For RGB pixels, set the corresponding indices in all three channels
                #input_lb[:, :, i // input_shape[3], i % input_shape[2]] -= self.epsilon
                #input_ub[:, :, i // input_shape[3], i % input_shape[2]] += self.epsilon
                input_lb[:, i // input_shape[-1], i % input_shape[-2]] -= self.epsilon
                input_ub[:, i // input_shape[-1], i % input_shape[-2]] += self.epsilon                    
        # flatten
        flatten_lb = input_lb.astype(np.float32).clip(0., 1.).reshape(-1,)
        flatten_ub = input_ub.astype(np.float32).clip(0., 1.).reshape(-1,)

        # set lower and upper bounds of inputs
        for i in range(self.flatten_inputVars.size):
            self.neural_network.setLowerBound(self.flatten_inputVars[i], flatten_lb[i])
            self.neural_network.setUpperBound(self.flatten_inputVars[i], flatten_ub[i])
            #if verbose > 1:
            #    if (flatten_lb[i] != flatten_ub[i]):
            #        print(f"Set {i} to [{flatten_lb[i]}, {flatten_ub[i]}]")

        # DEPRECATED: check if there are adversarial examples, inefficient since we need to check each class one by one
        # check if a y_i is larger than our label pred
        local_robust = True
        for y_idx, _ in enumerate(self.flatten_outputVars):
            if y_idx == self.out_label:
                continue
            # clear constraint of last class
            self.neural_network.additionalEquList.clear()
            # pred - y_i <= 0
            self.neural_network.addInequality([self.flatten_outputVars[self.out_label], self.flatten_outputVars[y_idx]], [1, -1], 0, isProperty=True)
            res, adv, _ = self.neural_network.solve(verbose=False, options=options)
            if res == 'unsat':
                # It is not possible that y_correct <= y_i, but just for this class
                # continue to check other classes
                if verbose == 2:
                    print(f"Cannot change the prediction from class {self.out_label} to class {y_idx}")
                continue
            else:
                # Either sat (we found an adversarial example) or timeout, 
                # if timeout, we assume that there are adversarial examples but we cannot find it
                if verbose == 2:
                    print(f"Can change the prediction from class {self.out_label} to class {y_idx}, or timeout")
                local_robust = False
                break
        ret =  (not local_robust)
        
        # self.neural_network.additionalEquList.clear()
        # self.neural_network.disjunctionList.clear()
        # new_disjuncts = []
        # for y_idx, _ in enumerate(self.flatten_outputVars):
        #     if y_idx == self.out_label:
        #         continue
        #     # pred - y_i <= 0
        #     new_disjunct = []
        #     scalar = 0.0
        #     eq = MarabouCore.Equation(MarabouCore.Equation.LE)
        #     eq.addAddend(1, self.flatten_outputVars[self.out_label])
        #     eq.addAddend(-1, self.flatten_outputVars[y_idx])
        #     eq.setScalar(scalar)    
        #     new_disjunct.append(eq)
        #     new_disjuncts.append(new_disjunct)
        # self.neural_network.addDisjunctionConstraint(new_disjuncts)
        
        # res, adv, _ = self.neural_network.solve(verbose=False, options=options)
        # ret = (res != 'unsat')
        
        return ret

if __name__ == "__main__":
#     # Network ACAS XU
#     # Information for the value to set in ../benchmarks/properties/acas_property_1.txt
    print('\n ACAS XU Network')
    # nn_filename = '../benchmarks/acasxu/ACASXU_experimental_v2a_1_1.nnet'
    nn_filename = 'Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_1_1.nnet'
    in_values = np.array([[[0.6, 0.0, 0.0, 0.0, 0.0]]])
    out_class = 1
    eps = 0.01
    orc_marabou = Oracle_marabou(nn_filename)
    orc_marabou.encode_aex((in_values, out_class), eps)#, image_type='others')
    ret = orc_marabou.has_aex(verbose=2)
    print("has AEx" if ret else "no AEx")

    # Network MNIST
    # Information for the value to set in ../benchmarks/properties/mnist/image1_target1_eps0.005.txt 
    print('\n MNIST Network')
    # nn_filename = '../benchmarks/mnist/mnist10x10.nnet'
    nn_filename = 'Marabou/resources/nnet/mnist/mnist10x10.nnet'
    in_values = np.array([0.0 for i in range(784)]).reshape((1, 28, 28))
    out_class = 5
    eps = 0.05
    orc_marabou = Oracle_marabou(nn_filename)
    orc_marabou.encode_aex((in_values, out_class), eps)#, image_type='grayscale')
    ret = orc_marabou.has_aex(verbose=1)
    print("has AEx" if ret else "no AEx")
    
    # Network maps 8x16 grayscale images two values
    print("\nConvolutional Network Example")
    # nn_filename = '../benchmarks/onnx/KJ_TinyTaxiNet.onnx'
    nn_filename = 'Marabou/resources/onnx/KJ_TinyTaxiNet.onnx'
    in_values = np.array([0.0 for i in range(8*16)]).reshape((1, 8, 16))
    out_class = 1
    eps = 0.1
    orc_marabou = Oracle_marabou(nn_filename)
    orc_marabou.encode_aex((in_values, out_class), eps)#, image_type='grayscale')
    ret = orc_marabou.has_aex(verbose=2)
    print("has AEx" if ret else "no AEx")

    print("\nCIFAR Network Example")
    # nn_filename = '../benchmarks/onnx/KJ_TinyTaxiNet.onnx'
    nn_filename = 'Marabou/resources/onnx/cifar10/cifar_base_kw_simp.onnx'
    orc_marabou = Oracle_marabou(nn_filename)
    in_values = np.ones((3, 32, 32))
    out_class = 2
    eps = 0.1
    orc_marabou.encode_aex((in_values, out_class), eps)#, image_type='rgb')
    ret = orc_marabou.has_aex(fixed=[i for i in range(900)], verbose=2)
    print("has AEx" if ret else "no AEx")
