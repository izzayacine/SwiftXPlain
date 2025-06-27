#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   MN-BaB Oracle
#
################################################################################
#import time
import sys
import os
#from comet_ml import Experiment  # type: ignore[import]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import torch
import numpy as np
from torch import nn
#from typing import Sequence, Tuple, List
from bunch import Bunch  # type: ignore[import]
from torch import Tensor

from .aex_oracle import AExOracle

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent/'mn-bab'))

from src.abstract_layers.abstract_network import AbstractNetwork
from src.mn_bab_verifier import MNBaBVerifier
from src.utilities.argument_parsing import get_args, get_config_from_json
from src.utilities.config import make_config, Dtype, Config
#from src.utilities.loading.data import transform_and_bound
from src.utilities.initialization import seed_everything
#from src.utilities.loading.data import transform_image, normalize
from src.utilities.loading.network import freeze_network, load_onnx_model
#from src.utilities.loading.network import load_net
#from src.utilities.logging import Logger, get_log_file_name
from src.verification_instance import VerificationInstance, generate_constraints
################################################################################


class Oracle_MNBaB(AExOracle):
    def __init__(self, nn_filename, gpu_id=0):
        super().__init__()
        
        seed_everything(0)
        
#         logger = Logger(sys.stdout)
#         sys.stdout = logger
#         logger.log_default(config)
        # torch is enable to detect available CPU, so we cannot use 2 threads for 2 diff CPUs
        torch.set_num_threads(1) # no multithreading when using python multiprocess resolution
        torch.cuda.empty_cache()
        #print('cuda activated?:', torch.cuda.is_available())
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = f'cuda:{gpu_id}' if torch.cuda.is_available() else f'cpu:{gpu_id}'
        
        # load the neural network
        if nn_filename.endswith('.onnx'):
            net_seq, onnx_shape, inp_name = load_onnx_model(nn_filename)
            network: nn.Module = net_seq
            input_dim = onnx_shape 
            # print('onnx shape:', onnx_shape)
#             if len(config.input_dim) == 0:
#                 print(f"Setting shape: {onnx_shape}")
#                 config.input_dim = onnx_shape                
        else:
            raise Exception('Unrecognized file format')

        torch.set_default_dtype(torch.float32)
        network = network.float()
        self.torch_nn = network.to(device)

        self.neural_network = AbstractNetwork.from_concrete_module(network, input_dim).to(device)
        freeze_network(self.neural_network)
        
        self.verifier = None
        
#         prima_dict = Bunch(sparse_n=50, K=3, s=1,
#                            num_proc_to_compute_constraints=2,
#                            max_unstable_nodes_considered_per_layer=1000,
#                            min_relu_transformer_area_to_be_considered=0.05,
#                            fraction_of_constraints_to_keep=1.0)
#         branch_dict = Bunch(method="babsr", 
#                             use_prima_contributions=False,
#                             use_optimized_slopes=False,
#                             use_beta_contributions=False,
#                             propagation_effect_mode="bias",
#                             use_indirect_effect= False,
#                             reduce_op="min",
#                             use_abs=True,
#                             use_cost_adjusted_scores=False)
#         config_dict = Bunch(n_layers= None,
#                        n_neurons_per_layer= None,
#                        input_dim= onnx_shape,
#                       eps=0.01,      
#                       random_seed=0,
#                       timeout=60,
#                       experiment_name='aex_mnbab',
#                       use_gpu=torch.cuda.is_available(),
#                       optimize_alpha = True,      
#                       optimize_prima = True,
#                       prima_hyperparameters = prima_dict, 
#                       branching = branch_dict,
#                       bab_batch_sizes = [1000, 1000, 1000],      
#                       comet_api_key="-",
#                       comet_project_name="-",
#                       comet_workspace="-",
#                       use_online_logging=False,
#                       recompute_intermediate_bounds_after_branching= True,
#                       normalization_means = 0.0,
#                       normalization_stds = 1.0,      
#                       device = torch_device     
#                       )
        root = str(Path(__file__).resolve().parent/'configs/mnbb')
        conv = '_conv' if 'conv' in nn_filename.lower() else ''
        if 'mnist' in nn_filename.lower():
            config_filename = root+f"/mnist{conv}.json"
        elif 'cifar10' in nn_filename.lower():
            config_filename = root+f"/cifar10{conv}.json"
        elif 'gtsrb' in nn_filename.lower():
            config_filename = root+f"/gtsrb{conv}.json"
        else:
            assert False, f'No json config file for {nn_filename}'
        
        config_dict = get_config_from_json(config_filename)
        config_dict.use_gpu = torch.cuda.is_available()
        config_dict.timeout = 60
        assert (tuple(config_dict.input_dim) == onnx_shape)
        self.config = make_config(**config_dict)
        

    def delete(self):
        """
            Delete the neural network and the robustness tool.
        """
        # self.neural_network.reset_input_bounds()
        # self.neural_network.reset_output_bounds()
        # self.neural_network.reset_optim_input_bounds()        
        pass
        #raise NotImplementedError("The delete_oracle() method is not implemented.")

        
    def encode_aex(self, instance, epsilon):
        """
            Parse the input instance and prepare constraints for finding adversarial examples.
        """      
        self.config.eps = epsilon
        self.config.verifier.verb = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.verifier = MNBaBVerifier(self.neural_network, device, self.config.verifier)

        self.in_values, self.out_label = instance
        # the perturbation bound is set in config file
        self.epsilon = epsilon
        
        sample = torch.from_numpy(instance[0]).type(torch.FloatTensor)
        sample = sample.unsqueeze(0).to(device) # increase dim => (1, c, w, h)
        
        net_out = torch.argmax(self.torch_nn(sample)).item()
        #self.probs =  self.torch_nn(sample)
        assert (self.out_label == net_out), f'target={self.out_label}, mn-bab output={net_out}'

        # clear constraint
        self.neural_network.reset_input_bounds()
        self.neural_network.reset_output_bounds()
        self.neural_network.reset_optim_input_bounds()


    def has_aex(self, fixed=[], timeout=60, verbose=0) -> bool:
        """
            Check if there exist adversarial examples.
        """
        self.config.timeout = timeout
        num_classes = self.neural_network.output_dim[-1]
        data_shape = self.in_values.shape
        assert (data_shape[0] in [1,3])
        n_pixels = data_shape[-1] * data_shape[-2]
        hypos = np.full((n_pixels,), False, dtype=bool)
        for i in fixed:
            hypos[i] = True 
        
        # reset input bounds
        self.neural_network.reset_input_bounds()
        self.neural_network.reset_output_bounds()
        self.neural_network.reset_optim_input_bounds()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set lower/upper bound +/- eps
        img = torch.from_numpy(self.in_values).type(torch.FloatTensor)
        img = img.to(device) 
        input_lb = torch.clamp(img - self.epsilon, 0., 1.)
        input_ub = torch.clamp(img + self.epsilon, 0., 1.)  
        
        for i in range(n_pixels):
            if hypos[i]:
                input_lb[:, i//data_shape[-1], i % data_shape[-2]] = img[:, i // data_shape[-1], i % data_shape[-2]]
                input_ub[:, i//data_shape[-1], i % data_shape[-2]] = img[:, i // data_shape[-1], i % data_shape[-2]]
        
        
        #img = image.view(image_dim)
        #input_lb = input_lb.view(image_dim)
        #input_ub = input_ub.view(image_dim)
        
        
        seed_everything(self.config.random_seed)
        target_gt_constraints = generate_constraints(num_classes, self.out_label)
        #print(self.probs[0])
        # sort classes to find AEx quicker
        #sorted(target_gt_constraints[0], key=lambda x: self.probs[0][x[0][1]].item(), reverse=True)
        aex_inst =  VerificationInstance(self.neural_network, self.verifier, self.config,
                                         [img], [(input_lb, input_ub)], target_gt_constraints,)

        # img = img.unsqueeze(0).to(device)
        # input_lb.unsqueeze(0).to(device)
        # input_ub.unsqueeze(0).to(device)   
        #aex_inst = VerificationInstance.create_instance_for_batch_ver(self.neural_network, self.verifier, 
        #                                                          img, input_lb, input_ub, 
        #                                                          self.out_label, self.config, 
        #                                                          num_classes)
        
        aex_inst.run_instance()
        local_robust = aex_inst.is_verified
        if verbose > 1:
            print('AEx:', aex_inst.adv_example[0])
        #======================
        if verbose: 
            if local_robust:   
                self.adv = None  
            elif aex_inst.adv_example is None:  
                # TO, unknown AEx
                self.adv = np.array([self.in_values+self.epsilon])                 
            else:
                self.adv = aex_inst.adv_example[0]
                if type(self.adv) == torch.Tensor:
                    self.adv = self.adv.numpy()  
        #=======================       
        aex_inst.free_memory()
        
        return (not local_robust)


if __name__ == "__main__":
    print('\n MNIST Network')
    nn_filename = '../benchmarks/onnx/mnist2x10.onnx'
    in_values = np.full((1, 28, 28), 0.5)
    out_class = 2
    
    eps = 0.05
    oracle = Oracle_MNBaB(nn_filename)
    oracle.encode_aex((in_values, out_class), eps)
    ret = oracle.has_aex(verbose=1)
    print("has AEx" if ret else "no AEx")