#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
#   alpha-beta-crown Oracle
#
################################################################################
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from .aex_oracle import AExOracle

#import time
#import onnx
import numpy as np
import copy
import re
import socket
import random
import gc
import torch
#import torch.nn.functional as F
from torch import nn

from pathlib import Path
sys.path.append(str(Path(__file__).resolve()))
#print(str(Path(__file__).resolve()))

from .crown.complete_verifier.abcrown import ABCROWN
from .crown.complete_verifier import arguments
#from loading import load_verification_dataset
from .crown.complete_verifier.specifications import SpecificationVerifiedAcc, construct_vnnlib
from .crown.complete_verifier.utils import Logger, print_model
#from loading import load_model_and_vnnlib, parse_run_mode, adhoc_tuning, Customized  # pylint: disable=unused-import
from .crown.complete_verifier.load_model import load_model #, load_model_onnx, Customized  # pylint: disable=unused-import
from .crown.complete_verifier.loading import parse_run_mode


#
#=========================================================#
class Oracle_abcrown(AExOracle):
    def __init__(self, nn_filename, gpu_id=0):        
        super().__init__()

        config_filename = ""
        root = str(Path(__file__).resolve().parent/'configs/abcrown')
        
        #conv = '_conv' if 'conv' in nn_filename.lower() else ''
        config_filename = root+f"/{os.path.basename(nn_filename).lower().split('.')[0]}.yaml"
        # if 'mnist' in nn_filename.lower():
        #     config_filename = root+f"/mnist{conv}.yaml"
        # elif 'cifar10' in nn_filename.lower():
        #     config_filename = root+f"/cifar10{conv}.yaml"
        # elif 'gtsrb' in nn_filename.lower():
        #     config_filename = root+f"/gtsrb{conv}.yaml"
        # else:
        #     assert False, f'No yaml config file for {nn_filename}'
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else f'cpu:{gpu_id}'
        args = ['--config', config_filename, '--device', device]
        
        self.crown = ABCROWN(args=args)
        arguments.Config = self.crown.get_config()
        #arguments.Config.parse_config(args)
        self.config =  arguments.Config     
        
        # main abcrown
        torch.manual_seed(arguments.Config['general']['seed'])
        random.seed(arguments.Config['general']['seed'])
        np.random.seed(arguments.Config['general']['seed'])
        torch.set_printoptions(precision=8)
        device = arguments.Config['general']['device']
        if device != 'cpu':
            torch.cuda.manual_seed_all(arguments.Config['general']['seed'])
            # Always disable TF32 (precision is too low for verification).
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if arguments.Config['general']['deterministic']:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
        if arguments.Config['general']['double_fp']:
            torch.set_default_dtype(torch.float64)
        if arguments.Config['general']['precompile_jit']:
            #precompile_jit_kernels()
            assert False, f'No precompile_jit_kernels option'

        bab_args = arguments.Config['bab']
        if bab_args['backing_up_max_domain'] is None:
            arguments.Config['bab']['backing_up_max_domain'] = bab_args['initial_max_domains']
        #_, _, _, _, self.neural_network, _, shape = parse_run_mode()
        #self.logger = Logger(run_mode, save_path, timeout_threshold)
        #self.logger = Logger(run_mode, save_path, timeout_threshold)

        # if arguments.Config['debug']['sanity_check']:
        #     arguments.Config['attack']['pgd_order'] = 'before'
        # load the neural network
        nn_filename = arguments.Config["model"]["path"]
        # Mahi: it is no more needed, since we are getting the nn model from Config
        # if nn_filename.endswith('.onnx'):
        #     arguments.Config["model"]["onnx_path"] = nn_filename
        #     arguments.Config['data']['start'] = 0 
        #     arguments.Config['data']['end'] = 1
        #     #self.neural_network = load_model()               
        # elif nn_filename.endswith('.model'):
        #     #pytorch format
        #     arguments.Config["model"]["path"] = nn_filename
        #     #self.neural_network = load_model()        
        # else:
        #     raise Exception('Unrecognized file format')                  
        #self.logger = Logger(run_mode, save_path, timeout_threshold)                 

        # if arguments.Config['debug']['sanity_check']:
        #     arguments.Config['attack']['pgd_order'] = 'before'
        # load the neural network
        nn_filename = arguments.Config["model"]["path"]
        # Mahi: it is no more needed, since we are getting the nn model from Config
        # if nn_filename.endswith('.onnx'):
        #     arguments.Config["model"]["onnx_path"] = nn_filename
        #     arguments.Config['data']['start'] = 0 
        #     arguments.Config['data']['end'] = 1
        #     #self.neural_network = load_model()               
        # elif nn_filename.endswith('.model'):
        #     #pytorch format
        #     arguments.Config["model"]["path"] = nn_filename
        #     #self.neural_network = load_model()        
        # else:
        #     raise Exception('Unrecognized file format')                  
        
        self.neural_network = load_model()
        print('model loaded')

    def delete(self):
        """
            Delete the neural network and the robustness tool.
        """
        pass

    def encode_aex(self, instance, epsilon):
        """
            Parse the input instance and prepare constraints for finding adversarial examples.
        """
        self.in_values, self.out_label = instance
        self.epsilon = epsilon

        # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.config['general']['device']
        self.neural_network.eval()
        # print('check nn.eval()')
        
        sample = torch.from_numpy(instance[0]).type(torch.FloatTensor)
        sample = sample.unsqueeze(0).to(device) # increase dim => (1, c, w, h)
        dnn = self.neural_network.to(device) # added by Mahi
        model_output = torch.argmax(dnn(sample)).item()
        print("Out_label: {0}".format(model_output))
        assert (self.out_label == model_output), f'target={self.out_label}, αβ-crown output={model_output}'
        
        arguments.Config['specification']['epsilon'] = self.epsilon

    
    def has_aex(self, fixed=[], timeout=60, verbose=0) -> bool:
        """
            Check if there exist adversarial examples.
        """        
        data_shape = self.in_values.shape
        num_classes = self.config['data']['num_outputs']
        assert (data_shape[0] in [1,3])
        n_pixels = data_shape[-1] * data_shape[-2]
        hypos = np.full((n_pixels,), False, dtype=bool)
        for i in fixed:
            hypos[i] = True

        input_lb = input_ub = None
        device = self.config['general']['device']

        # set lower/upper bound +/- eps
        img = torch.from_numpy(self.in_values).type(torch.FloatTensor)
        img = img.to(device)
        input_lb = torch.clamp(img - self.epsilon, 0., 1.)
        input_ub = torch.clamp(img + self.epsilon, 0., 1.)

        for i in range(n_pixels):
            if hypos[i]:
                input_lb[:, i//data_shape[-1], i % data_shape[-2]] = img[:, i // data_shape[-1], i % data_shape[-2]]
                input_ub[:, i//data_shape[-1], i % data_shape[-2]] = img[:, i // data_shape[-1], i % data_shape[-2]]

        if data_shape[0] == 3:
            # 3 channels (rgb image)
            input_lb, input_ub = input_lb.unsqueeze(0), input_ub.unsqueeze(0)
            
        
        logger = Logger('swiftXP', 'out.txt', timeout)
        logger.record_start_time()
        self.config['bab']['timeout'] = float(timeout) # use the timeout from the input
        
        x_range = torch.stack([input_lb.flatten(1), input_ub.flatten(1)], -1).detach().cpu().numpy()
        dic_vnnlib = {'labels': torch.tensor([self.out_label])}
        sva = SpecificationVerifiedAcc()
        vnnlib = sva.construct_vnnlib(dic_vnnlib, x_range, [0])[0]
        # np.save('robxpl_x_range.npy', x_range)
        #
        # Mahi: the following part of code was added to write the output vnnlib to a file
        # import pickle
        # with open('robxpl_vnnlib.pkl', 'wb') as f:
        #     pickle.dump(vnnlib, f)
        # vnnlib = construct_vnnlib({'X': img, 'eps': 0.0, 'labels': torch.tensor([self.out_label])}, [0])[0]
        # print(dic_vnnlib, x_range)
        #shape = arguments.Config['model']['input_shape']
        #Mahi: NOTE the following assertion fails
        # assert (shape == data_shape)
                
        verif_status = self.crown.verify(self.neural_network, vnnlib, (-1,)+data_shape, logger)

        if verbose > 1:
            # Summarize results.
            logger.summarize_results(verif_status, 0)

        logger.finish(verb=verbose > 1)
        # veri_ret = logger.verification_summary.keys()
        
        self.adv = None
        local_robust = True
        if verif_status in ('verified', 'safe', 'unsat', 'safe-mip', 'safe-incomplete', 'safe-incomplete-refine'):
            local_robust = True
        elif verif_status in ('falsified', 'unsafe', 'attack success', 'unsafe-mip', 'unsafe-pgd', 'unsafe-bab'):
            local_robust = False
            self.adv = np.array([self.in_values+self.epsilon])
        elif verif_status in ('unknown',  'timeout', 'unknown-mip'):
            local_robust = False
        else:
            raise ValueError('Unknown verification result')        

        return (not local_robust)

if __name__ == "__main__":
    # print("\nCIFAR Network Example")
    # in_values = np.full((3, 32, 32), 0.5)
    # out_class = 0
    
    # eps = 0.25
    # abcrown_oracle = Oracle_abcrown(args=sys.argv[1:])
    # abcrown_oracle.encode_aex(instance=(in_values, out_class), epsilon=eps)
    # ret = abcrown_oracle.has_aex(fixed=list(range(100,2000)), timeout=600, verbose=0)
    # print("has AEx" if ret else "no AEx")

    # print("\nMNIST Network Example")
    # in_values = np.random.rand(1, 28, 28)
    # #in_values = np.zeros((1, 28, 28), dtype=float)
    # out_class = 5
    # eps = 0.3
    # abcrown_oracle = Oracle_abcrown(args=sys.argv[1:])
    # abcrown_oracle.encode_aex(instance=(in_values, out_class), epsilon=eps)
    # ret = abcrown_oracle.has_aex(fixed=[0], timeout=300, verbose=1)
    # print("has AEx" if ret else "no AEx")

    print("\nCIFAR Network Example")
    in_values = np.random.rand(3, 32, 32)
    out_class = 2
    eps = 2./255.
    abcrown_oracle = Oracle_abcrown(args=sys.argv[1:])
    abcrown_oracle.encode_aex(instance=(in_values, out_class), epsilon=eps)
    ret = abcrown_oracle.has_aex(fixed=[0], timeout=300, verbose=1)
    print("has AEx" if ret else "no AEx")    

# python aex_abcrown.py --config ./configs/mnist_cnn_a_adv.yaml 
# add --device cpu with the above command 
