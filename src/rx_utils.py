
import onnx
import onnxruntime 

from timeit import default_timer as timer

import torch
#import torchvision
#import torchvision.transforms as transforms
#import torch.nn.functional as F
import numpy as np

from scipy.special import softmax
from skimage.color import gray2rgb, rgb2gray
from matplotlib import pyplot as plt

#
#========================================================================
TIME_LIMIT = 120

#========================================================================
def splitChnk(length, top_id, dist='uniform'):
    if dist == 'uniform':
        k = max(int(length*0.05), 1) # split into 5% of total size
        k0 = 0 if (top_id>0) else k
        chunks = [j+top_id for j in range(k0, length+1, k)]
        if chunks[-1] < length+top_id:
            chunks += [length+top_id]            
    elif dist == 'geometric':
        assert not top_id 
        z = np.random.geometric(p=0.3, size=1000)
        pr = [(np.count_nonzero(z == i)/1000) for i in np.unique(z)]
        chunks = [int(k*length) for k in pr[-1] if int(k*length)>0]
        chunks = [k+chunks[:i] for i,k in enumerate(chunks)]
        assert (chunks[-1] < length)
        chunks +=[length]
    else:
        raise NotImplementedError('Not implemeted distribution')
        
    return chunks

def binsplit(lb, ub, n):
    x = [lb, ub]
    while n+2 > len(x):
        x2 = []
        for i in range(len(x)-1):
            if len(x2)+len(x[i:]) == n+2:
                x2 = x2+x[i:]
                break
            mid = (x[i]+x[i+1])//2
            x2.append(x[i])
            if mid > x[i]:
                x2.append(mid)
            if i+1 == len(x) - 1:    
                x2.append(x[i+1])
        if len(x) == len(x2):
            break
        x = x2
    if len(x) == n+2:
        x = x[1:-1] # rm lb,ub
    elif len(x) == n+1:
        x = x[:-1] # rm ub
    return x

#
#========================================================================
def neg_advx_ok(res):
    unsat = False
    aexok = False
    ae_k = 0 # last AEx
    for i,x in enumerate(res):
        if x.value == 0:
            unsat = True
            break
        elif x.value == 1:
            aexok = True
            ae_k = i

    return (aexok and unsat and (ae_k==i-1)) or (aexok and (ae_k+1==len(res))) or (unsat and i==0)

#====================

def advx_ok(res):
    aex_no = False
    aex_ok = False
    k = 0 # last ¬AEx
    for i,x in enumerate(res):
        if x.value == 0:
            aex_no = True
            k = i
        elif x.value == 1:
            aex_ok = True
            break
    # [ ...,0,1,...]
    # [ ....,0 ] [1]
    # [0] [ 1,.... ]
    return (aex_ok and aex_no and (k == i-1)) or (aex_no and (k+1==len(res))) or (aex_ok and i==0)    

#
#========================================================================
#

def run_solver(oracle, hypos, res=None, cond=None):
    """
        Run a single solver on a given formula/assumptions.
    """
    t0 = timer()
    #logging.info(f"starting {solver} for {cid+1} chunks")
    aex = oracle.has_aex(fixed=hypos, timeout=TIME_LIMIT)
    #logging.info(f"finished {solver} for {cid+1} chunks -- {res} outcome")
    res.value = int(aex)
    time = timer() - t0
    #with open('/home/users/nus/dcsv206/log.txt', 'a') as fp:
    #    fp.write(f'{time}\n')        
    if cond is not None:
        with cond:
            cond.notify()
    
    return aex

# def run_solver2(oracle, hypos, res, cond, start_flag, end_flag, idx):
#     """
#         Run a single solver on a given formula/assumptions.
#     """
#     while True:
#         if end_flag.is_set():
#             break
#         #print('wait', idx)    
#         start_flag.wait()
#         #print('start: ',idx) 
#         x = np.array(hypos)
#         hypos2 = x[(x>0)]  
#         aex = oracle.has_aex(fixed=hypos2, timeout=TIME_LIMIT)
#         res.value = int(aex)
#         start_flag.clear()
#         #print('notify ', idx)
#         if cond is not None:
#             with cond:
#                 cond.notify()
#     print('terminate',idx)
#     return aex
    

def run_solver3(vnn_slv, inst, eps, hypos, res=None, cond=None):
    """
        Run a single solver on a given (inst, eps, fixed_features).
    """
    t0 = timer()
    
    oracle =  vnn_slv.get_oracle(name='mnbab')
    oracle.encode_aex(inst, eps)
    
    aex = oracle.has_aex(fixed=hypos, timeout=TIME_LIMIT)
    
    res.value = int(aex)
    time = timer() - t0

    if cond is not None:
        with cond:
            cond.notify()
    
    return aex

def run_solver4(queue, res_q, gpu_id, vnn_slv):
    torch.cuda.set_device(gpu_id)

    oracle =  vnn_slv.get_oracle()
    encoded = False

    print(f"[GPU {gpu_id}] Worker started.")
    # model = torch.load(model_path, map_location=f"cuda:{gpu_id}")

    while True:
        task = queue.get()
        if task == "STOP":
            print(f"[GPU {gpu_id}] Received STOP signal. Exiting.")
            break

        inst, eps, hypos = task
        if not encoded:
            oracle.encode_aex(inst, eps)
            encoded = True

        #print(f"[GPU {gpu_id}] Running ...")

        # Example: Run your real verifier here
        # result = your_verifier_function(model, input_data.to(f"cuda:{gpu_id}"))
        try:
            aex = oracle.has_aex(fixed=hypos, timeout=TIME_LIMIT)
        except Exception as e:
            aex = -1    
        res_q.put(int(aex))
        # try:
        #     res_q.put(int(aex), timeout=TIME_LIMIT)
        # except Full:
        #     pass  
        
        # print(f"[GPU {gpu_id}] Done, aex= {aex}")

#==============================================================================

class DNN(object):
    
    def __init__(self, filename):
        self.sess_opt = onnxruntime.SessionOptions()
        self.sess_opt.intra_op_num_threads = 3
        self.sess_opt.inter_op_num_threads = 3
        self.sess_opt.execution_mode  = onnxruntime.ExecutionMode.ORT_PARALLEL 
        #self.sess_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL 
        self.sess_opt.add_session_config_entry('session.intra_op_thread_affinities', '1;2')
        self.sess_opt.add_session_config_entry('session.inter_op_thread_affinities', '1;2')     
        # graph = onnx.load(filename).graph
        self.filename = filename

    def __del__(self):
        del  self.sess_opt
        del self 

    def predict(self, data):    
        return np.argmax(self.__call__(data), axis=1) 
    
    def __call__(self, data):
        if self.filename.endswith('.onnx'):
            logits=  self.run_onnx(data)
        else:
            raise NotImplementedError('run .nnet not implemented')
        
        probs = softmax(logits, axis=1)
        return probs
    
    def lime_rgb_predict(self, data):
        inputTensor = data.astype(np.float32)
        if data.shape[-1] == 3:
            # reoder channel axis
            assert(len(data.shape) == 4) # batch
            inputTensor = np.moveaxis(inputTensor, -1, 1)
        if self.filename.endswith('.onnx'):
            logits = self.run_onnx(inputTensor)
        else:
            raise NotImplementedError('run .nnet not implemented')
        probs = softmax(logits, axis=1)
        return probs

    def lime_predict(self, data):
        inputTensor = data.astype(np.float32)
        if data.shape[-1] == 3:
            # convert rgb to gray
            inputTensor = rgb2gray(inputTensor) 
            inputTensor = np.expand_dims(inputTensor, axis=1)

        if self.filename.endswith('.onnx'):
            logits = self.run_onnx(inputTensor)
        else:
            raise NotImplementedError('run .nnet not implemented')
        probs = softmax(logits, axis=1)
        return probs           

    def run_onnx(self, data):
        sess = onnxruntime.InferenceSession(self.filename, self.sess_opt)
        #inputTensor = data.astype(np.float32)
        #logits = sess.run(None, {"input": inputTensor})[0]
        logits = []
        inputTensor = data.astype(np.float32)
        inputTensor = np.split(inputTensor, len(data))
        for x in inputTensor:
            logits.append(sess.run(None, {"input": x})[0][0])
        logits = np.array(logits)
        #logits = sess.run(None, {"input": inputTensor})[0]            
        #probs = F.softmax(logits, dim=1)
        return logits


def save_figure(image, path, cmap=None):
    """
    To plot figures.
    :param image: the image array of shape (width, height, channel)
    :param path: figure name.
    :param cmap: 'gray' if to plot gray scale image.
    :return: an image saved to the designated path.
    """
    fig = plt.figure()
    ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if cmap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)       