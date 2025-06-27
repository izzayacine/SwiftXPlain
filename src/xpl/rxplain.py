#!/usr/bin/env python3
#-*- coding:utf-8 -*-


#import subprocess
import sys
import os
import resource
import collections
from tqdm import tqdm
import warnings

from timeit import default_timer as timer

import numpy as np
from six.moves import range
import six
import math

#import torch
#from scipy.special import softmax

#import logging
# import multiprocessing as mp
import torch.multiprocessing as mp
#from  concurrent.futures import ProcessPoolExecutor
#from concurrent.futures import TimeoutError
#import concurrent.futures as cf


# import lime
# from lime import lime_image
# from lime.wrappers.scikit_image import SegmentationAlgorithm
# from skimage.color import gray2rgb, rgb2gray

from pysat.examples.hitman import Hitman

from pathlib import Path
import warnings
# try:
#     from aex.aex_marabou import Oracle_marabou
# except ImportError:
#      warnings.warn("Marabou is unavailable")
try:        
    from aex.aex_mnbab import Oracle_MNBaB
except ImportError:
     warnings.warn("mn-bab is unavailable")
try:
    from aex.aex_abcrown import Oracle_abcrown
except ImportError:
     warnings.warn("Crown is unavailable")        

from rx_utils import DNN, run_solver4

from .axp import AXp
from .cxp import CXp

#========================================================================

XConfig = collections.namedtuple('XConfig', ['alg', 'featD', 'approx', 'sort', 'slv']) 

#========================================================================

class RobXplainer():
    """
        Robustness reasonner -based computing explanations.
    """
    #tlimit = 120 # time-limit
    #mx_proc = mp.cpu_count()
    #NUM_GPUS = torch.cuda.device_count()
    
    def __init__(self, filename, config, ncpu=16, verb=1):
        #self.nn_fname = filename
        self.vnn_slv = NeuralNetVerifier(filename)
        self.oracles = [] 
        self.opts = config
        self.mx_proc = ncpu #if(ncpu < NUM_GPUS) else NUM_GPUS
        self.mxp2 = int(0.19*ncpu) if self.mx_proc > 30  else int(0.4*ncpu)
        if 'mnist' in filename.lower() and not('conv' in filename.lower()):
           self.mxp2 = 5
           #self.mxp2 = 4
        if 'gtsrb' in filename.lower():
            self.mxp2 = int(0.7*self.mx_proc) # 21     
        
        # fork vs spawn
        # mp.set_start_method('fork') # Multi-CPU faster
        mp.set_start_method('spawn', force=True) # Nvidia GPUs
        
        # heuristic explainer  
        # self.hexp = lime_image.LimeImageExplainer()
        self.hexp = None
        self.dnn = DNN(filename) 
        
        self.verbose = verb
        
    def __del__(self):
        # delete solvers
        for o in self.oracles:
            o.delete()
        del self.oracles    
        # delete lime    
        del self.hexp
        # del onnx/nnet dnn
        del self.dnn    

    def has_aex(self, hypos, gpu_id=0):
        task = (self.inst, self.eps, hypos)
        self.tqueues[gpu_id].put(task)
        aexok = self.res_q[gpu_id].get()
        # try:
        #     aexok = self.res_q[gpu_id].get(timeout=TIME_LIMIT)
        # except Empty:
        #     aexok = -1     
        return aexok

    def has_aex3(self, hypos):
        aexok = mp.Value('i', -1)
        proc = mp.Process(target=run_solver3, args=(self.vnn_slv, self.inst, self.eps, hyopos, aexok))
        proc.start()
        proc.join()  
        return aexok.value          

    def explain(self, inst, eps=0.08, xtype='abd', optim=False):
        """
            Compute a set of literals responsible of the prediction.
        """
                    
        sample, label = inst
        self.inst = inst  
        self.eps = eps
        if not len(self.oracles):
            #self.oracles = [self.vnn_slv.get_oracle(self.opts.slv) for i in range(self.mx_proc)] # incremental mode
            self.oracles = [self.vnn_slv.get_oracle(self.opts.slv)]
        print("oracle:", self.opts.slv)


        #self.assums = np.arange(0, len(sample.flatten())) # 1 channel images
        self.assums = np.arange(0, sample.shape[-1]*sample.shape[-2]) # 1/3 channels
        for oracle in self.oracles:
            oracle.encode_aex(inst, eps)
        self.slv = self.oracles[0]
        
        
        self.order = np.arange(len(self.assums))
        print('heuristic:', f'{"Sensitivity" if self.opts.sort else "lexico"}')
        if self.opts.sort:
            self.order = self.sensitivity_img(inst)
        # else:    
        #     if not self.hexp:
        #         self.hexp = lime_image.LimeImageExplainer()
        #     _, imprt = self.lime_explain(inst)
        #     self.order = np.argsort(-imprt, axis=None) # flatten, descending            
            
        
        #self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
        #        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        self.time = timer()            
        
        if self.verbose:
            assert(not self.slv.has_aex(self.assums))
            #assert(self.slv.has_aex())
        
        if not self.slv.has_aex(verbose=1):
            print("epsilon {0} is too small".format(self.eps))
            return []
        
        if xtype == 'abd':
            # abductive (PI-) explanation
            print ("**** Compute an AXp")
            axp = AXp(self, self.verbose) 
            xpl = eval('axp.'+self.opts.alg)
        else:
            # contrastive explanation
            print ("**** Compute a CXp")
            cxp = CXp(self, self.verbose)
            xpl = eval('cxp.'+self.opts.alg)

            # approx (w)cxp
            assert (self.slv.adv is not None)
            imgdata = inst[0].reshape(self.slv.adv.shape)
            diff = np.absolute(self.slv.adv - imgdata)
            diff = diff / eps
            if diff.shape[1] == 3:
                diff = np.transpose(np.squeeze(diff))            
                diff =  np.dot(diff[...,:3],[0.2989, 0.5870, 0.1140])
                diff = np.transpose(diff)
            diff[diff > 0.0] = 1
            diff = (np.ceil(diff)).astype(int)

            x = diff.flatten()*self.order
            x = np.ma.masked_equal(x, 0.)
            if self.opts.approx and not optim:
                self.order = x[x.mask == False]            
        
        if self.verbose:
            print('hypos len:', len(self.order))
        hypos = [self.assums[i] for i in self.order]

        # init (spawn) processes/oracles
        processes = []
        # One queue per GPU process    
        self.tqueues = [mp.Queue(maxsize=2) for _ in range(self.mx_proc)] # call GPU or STOP worker
        self.res_q = [mp.Queue(maxsize=1) for _ in range(self.mx_proc)]
        # Start GPU worker processes
        for gpu_id in range(self.mx_proc):
            p = mp.Process(target=run_solver4, args=(self.tqueues[gpu_id], self.res_q[gpu_id], gpu_id, self.vnn_slv))                                    
            p.start()
            processes.append(p)  

        if optim:
            if xtype == 'abd':
                core = self._small_axp(hypos, unit_sz=True)        
            else: 
                assert xtype == 'con' 
                core = self._small_cxp(hypos)
        else:        
            core = xpl(hypos)
        
        expl = sorted(core)
        assert len(expl)

        ## for o in self.oracles:
        ##     o.delete()
        
        # Send STOP to all GPU queues
        for q in self.tqueues:
            q.put("STOP")
        # Wait for all to finish
        for p in processes:
            p.join()   
        print(" All GPU processes finished.")
        
        self.time = timer() - self.time

        if self.verbose:
            print("expl-selctors: ", expl)
            print('expl len:', len(expl))
            print('expl time: {0:.3f}'.format(self.time))
            print('---------')
            print(" ncalls:", self.ncalls)
            print(" eps:", eps)             

        return expl

    # def main(self, inst=([],0), eps=0.08, xtype='abd', optim=False):
    #     self.inst = inst  
    #     self.eps = eps

    #     mp.set_start_method("spawn", force=True)
    #     import time
    #     NUM_GPUS = self.mx_proc
    #     N_ITER = 3  # number of iterations
    #     VERIFIERS = [f"slv{i}" for i in range(NUM_GPUS)]

    #     # Simulate input data and model
    #     model_path = "dummy_model.pth"  # Pretend path
    #     input_data = "dummy_input"      # Replace with real data if needed

    #     # One queue per GPU process
    #     queues = [Queue() for _ in range(NUM_GPUS)]
    #     processes = []
    #     res = [mp.Value('i', -1) for _ in range(NUM_GPUS)]

    #     # Start GPU worker processes
    #     for gpu_id in range(NUM_GPUS):
    #         p = mp.Process(target=worker_process, args=(queues[gpu_id], gpu_id, res[gpu_id]))
    #         p.start()
    #         processes.append(p)

    #     # Dispatch jobs to each GPU queue per iteration
    #     for iter_id in range(N_ITER):
    #         print(f"\n>>>>> Iteration {iter_id + 1}")
    #         for gpu_id in range(NUM_GPUS):
    #             task = (VERIFIERS[gpu_id], model_path, input_data)
    #             queues[gpu_id].put(task)
    #         time.sleep(2)
    #     print('\n>>>>end of itations\n')  
    #     print([v.value for v in res])  
    #     # Send STOP to all GPU queues
    #     for q in queues:
    #         q.put("STOP")

    #     # Wait for all to finish
    #     for p in processes:
    #         p.join()

    #     print(" All GPU processes finished.")
    #     print([v.value for v in res])        
    
    def _small_cxp(self, hypos, unit_sz=False):
        """
            Implicit hitting set algo to compute smallest/minimum CXp
        """    
        expl = []
        assert (len(self.assums) == len(hypos))
        to_hit = [i for i in range(len(hypos))]
        h2id = {h:i for i,h in enumerate(hypos)}
        
        with Hitman(bootstrap_with=[to_hit], htype='sorted') as hitman:
            # compute unit-size MCS/CXp
            if unit_sz:
                for i in range(len(hypos)):
                    if self.slv.has_aex(hypos[:i]+hypos[i+1:]):
                        expl = [hypos[i]]
                        break
            # compute unit-size MUSs/AXps
            # TODO
            
            # main loop
            ncalls = 0
            while True:
                hset = hitman.get()
                # loop must stop before exhaustive exploration
                assert (hset is not None)             
                if self.verbose > 1:
                    print('\nhset:', [hypos[i] for i in hset]) 
                if self.verbose:
                    print('\n#hset:',len(hset))    
                ncalls += 1

                to_test = [h for i,h in enumerate(hypos) if (i not in hset)]    
                if not self.slv.has_aex(to_test):
                    # no AEx, reduce waxp and block
                    xpl = AXp(self)
                    xpl = eval('xpl.'+self.opts.alg)
                    core = xpl(to_test)
                    to_hit = [h2id[h] for h in core]
                    hitman.hit(to_hit)
                    if self.verbose:
                        print('#to_test=', len(to_test), ' #to_hit=',len(to_hit))
                        print('ncalls:', ncalls)
                else:
                    expl = [hypos[i] for i in hset]
                    break
        self.ncalls = ncalls
        return expl

    def _small_axp(self, hypos, unit_sz=False):
        """
            Implicit hitting set algo to compute smallest/minimum AXp
        """    
        expl = []
        assert (len(self.assums) == len(hypos))
        to_hit = [i for i in range(len(hypos))]
        h2id = {h:i for i,h in enumerate(hypos)}
        
        with Hitman(bootstrap_with=[to_hit], htype='sorted') as hitman:
            # compute unit-size MCS/CXp
            if unit_sz:
                for i in range(len(hypos)):
                    if self.slv.has_aex(hypos[:i]+hypos[i+1:]):
                        hitman.hit([hypos[i]])
            
            # main loop
            ncalls = 0
            while True:
                hset = hitman.get()
                # loop must stop before exhaustive exploration
                assert (hset is not None)             
                if self.verbose > 1:
                    print('\nhset:', [hypos[i] for i in hset]) 
                if self.verbose:
                    print('\n#hset:',len(hset))    
                ncalls += 1

                
                if self.slv.has_aex([hypos[i] for i in hset], verbose=1):
                    # AEx, reduce wcxp and block
                    xpl = CXp(self)
                    xpl = eval('xpl.'+self.opts.alg)
                    # TODO extract unsat core from AEx
                    to_test = [h for i,h in enumerate(hypos) if (i not in hset)]
                    core = xpl(to_test)
                    to_hit = [h2id[h] for h in core]
                    hitman.hit(to_hit)
                    if self.verbose:
                        print('#to_test=', len(to_test), ' #to_hit=',len(to_hit))
                        print('ncalls:', ncalls)
                else:
                    expl = [hypos[i] for i in hset]
                    break
        return expl


    def enumerate(self, inst, eps=0.08, xnum=100, unit_sz=False):
        """
            MARCO algo to enumerate AXp/CXp
        """
        sample, label = inst
        self.inst = inst  
        self.eps = eps
        if not len(self.oracles):
            #self.oracles = [self.vnn_slv.get_oracle(self.opts.slv) for i in range(self.mx_proc)] # incremental mode
            self.oracles = [self.vnn_slv.get_oracle(self.opts.slv)]
        print("oracle:", self.opts.slv)
        
        #self.assums = np.arange(0, len(sample.flatten())) # 1 channel images
        self.assums = np.arange(0, sample.shape[-1]*sample.shape[-2]) # 1/3 channels
        for oracle in self.oracles:
            oracle.encode_aex(inst, eps)
        self.slv = self.oracles[0]
        
        #order = np.arange(len(self.assums))
        #if self.opts.sort:
        order = self.sensitivity_img(inst)
        hypos = [self.assums[i] for i in order]
        
        time = timer()            
        if self.verbose:
            assert(not self.slv.has_aex(self.assums))
        
        if not self.slv.has_aex():
            print("epsilon {0} is too small".format(self.eps))
            return []
        
        # === MARCO enum ==== #
        print ("*** Enum  AXp/CXp ***")
        #to_hit = [i for i in range(len(hypos))]
        h2id = {h:i for i,h in enumerate(hypos)}
        cxps, axps = [], []
        ffa = np.zeros(len(hypos))
        
        with Hitman(bootstrap_with=[self.assums.tolist()], htype='lbx') as hitman:
            # compute unit-size MCS/CXp
            if unit_sz:
                for i in range(len(hypos)):
                    if self.slv.has_aex(hypos[:i]+hypos[i+1:]):
                        hitman.hit([hypos[i]])
                        cxps.append([hypos[i]])
            
            # main loop
            ncalls = 0
            hset = hitman.get()
            while hset is not None:
                if self.verbose>1:
                    print('\n#hset:',len(hset))    
                if ncalls >= xnum:
                    break
                ncalls += 1
                
                to_test = [hypos[i] for i in hset]
                if self.slv.has_aex(to_test, verbose=1):
                    # AEx, compute cxp 
                    xpl = CXp(self, verb=0)
                    xpl = eval('xpl.'+self.opts.alg)
                    # TODO extract unsat core from AEx
                    to_test = [h for i,h in enumerate(hypos) if (i not in hset)]
                    expl = xpl(to_test)
                    to_hit = [h2id[h] for h in expl]
                    hitman.hit(to_hit)
                    if self.verbose:
                        print('cxp:', expl) 
                    cxps.append(expl)       
                else:
                    # ¬AEx, compute axp
                    xpl = AXp(self, verb=0)
                    xpl = eval('xpl.'+self.opts.alg)
                    expl = xpl(to_test)
                    blk = [h2id[h] for h in expl] 
                    hitman.block(blk)
                    if self.verbose:
                        print('axp:', expl) 
                    axps.append(expl)                              
                # report expl    
                yield (expl)
                hset = hitman.get()
                # ffa score
                for i in expl:
                    ffa[i] += 1./len(expl)                 
        time = timer() - time
        expl = np.where(ffa > 0.)[0].tolist()
        lnc = sum([len(x) for x in cxps])/len(cxps) if len(cxps) else 0 
        lna = sum([len(x) for x in axps])/len(axps) if len(axps) else 0
        if self.verbose:
            print()
            print('#cxp:', len(cxps))
            print('#axp:', len(axps))
            print('avg cxp:', lnc)
            print('avg axp:', lna)
            print('---------')
            print('ffa score:', ffa[ffa > 0.].tolist())
            print("expl-selctors: ", expl)
            print('expl len:', len(expl))            
            print('expl time: {0:.3f}'.format(time))
            print('---------')
            print(" ncalls:", ncalls)
            print(" eps:", eps)         
            self.ffa = ffa

    def lime_explain(self, inst):
        """
        explaining images using LIME
        """
        
        imgdata, pred = inst
        image2 = np.squeeze(imgdata)
        if (image2.shape[0]== 3):
            pred_fn = self.dnn.lime_rgb_predict
            image2 = np.transpose(image2) # np.moveaxis(image2, 0, -1)
        else:
            pred_fn =  self.dnn.lime_predict
            image2 =  gray2rgb(image2)

        #batch = np.expand_dims(imgdata, axis=0)
        #pred = self.dnn.predict(batch)
        
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        #segmenter = SegmentationAlgorithm('slic', n_segments=15)
        
        explanation = self.hexp.explain_instance(image2, classifier_fn=pred_fn,
                                                 top_labels=10, hide_color=0, 
                                                 num_samples=1000, segmentation_fn=segmenter)
        
        exp = explanation.local_exp[pred]
        
        seg_min = np.min(explanation.segments)
        if seg_min > 0:
            explanation.segments = explanation.segments - seg_min
        segments = explanation.segments
        seg2imprt = {seg: imprt for seg, imprt in exp}
        lit2imprt = {}
        
#         flat_img = imgdata.flatten()
#         flat_seg = segments.flatten()
#         for i, (x, seg) in enumerate(zip(flat_img, flat_seg), start=1):
#             imprt = seg2imprt[seg]
#             lit = i if x > 0 else -i
#             lit2imprt[lit] = imprt
        
        return seg2imprt, segments    

    def sensitivity_img(self, inst):
        """
        sensitivity map.
        """
        imgdata, label = inst
        image = np.transpose(imgdata)
        #width, height, channel = image.shape[0], image.shape[1], image.shape[2]
        channel, width, height = imgdata.shape[0], imgdata.shape[1], imgdata.shape[2]
        
        temp = image.reshape(width * height, channel)
        image_batch = np.kron(np.ones(shape=(width * height, 1, 1), dtype=temp.dtype), temp)
        image_batch_manip = image_batch.copy()
        for i in range(width * height):
            if channel == 1:
                image_batch_manip[i][i][:] = 1 - image_batch_manip[i][i][:]
            else:
                #assert (channel == 3)
                image_batch_manip[i][i][:] = 0

        image_batch = image_batch.reshape((width * height, width, height, channel))
        image_batch = np.transpose(image_batch, (0,3,2,1))
        predictions = self.dnn.run_onnx(image_batch)
        #predictions = np.asarray(predictions[0])
        image_batch_manip = image_batch_manip.reshape((width * height, width, height, channel))
        image_batch_manip = np.transpose(image_batch_manip, (0,3,2,1))
        predictions_manip = self.dnn.run_onnx(image_batch_manip)
        #predictions_manip = np.asarray(predictions_manip[0])
        difference = predictions - predictions_manip
        features = difference[:, label]
        sorted_index = features.argsort()
        #sensitivity = features.reshape(width, height)

        return sorted_index
               


#====================================================================#

class NeuralNetVerifier(object):
    """
        Factory class to create AEx oracle
    """
    
    def __init__(self, nn_filename):
        self.nnet = nn_filename
        
    def get_oracle(self, name='crown'):
        if name == 'marabou':
            return Oracle_marabou(self.nnet)
        elif name == 'mnbab':
            return  Oracle_MNBaB(self.nnet)         
        elif name == 'crown':
            return  Oracle_abcrown(self.nnet)  
        else:
            raise Exception('There is no {0} robustness verifier!'.format(name))    
            #
            
