#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## xxxx.py
##
##  Created on: June 13, 2023
##

#import subprocess
import sys
import os
import random
import resource
import getopt
import collections
from pathlib import Path


from six.moves import range
import six
import math
import numpy as np

#import torch
#print('cuda activated?:', torch.cuda.is_available())

from skimage.color import label2rgb

from xpl.rxplain import RobXplainer, XConfig
from rx_utils import save_figure


#
#==============================================================================
def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'a:c:d:e:ho:sSx:X:v',
                                   ['alg=',
                                    'cpu=',
                                    'delta=',
                                    'eps=',
                                    'enum=',
                                    'FD=',
                                    'help',
                                    'oracle=',
                                    'sort',
                                    'min',
                                    'sample=',
                                    'xtype=',
                                    'verb'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    # init 
    verb = 0
    xtype = 'abd'
    alg = 'swift'
    oracle = 'marabou'
    sample = None
    eps = 0.05
    ncpu = 16
    featD = 0.75
    sorting = True
    smallest = False
    xnum = 1

    for opt, arg in opts:
        if opt in ('-a', '--alg'):
            alg = str(arg)
            assert (alg in ['swift', 'linear', 'dicho'])
        elif opt in ('-c', '--cpu'):
            ncpu = int(arg)
            # NUM_GPUS = torch.cuda.device_count()
            # ncpu = ncpu if(ncpu < NUM_GPUS) else NUM_GPUS 
        elif opt in ('--enum'):
            xnum = int(arg)             
        elif opt in ('--FD'):
            featD = float(arg)
            assert 0. <= featD <= 1.                        
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-o', '--oracle'):
            oracle = str(arg) 
            assert (oracle in ['marabou', 'mnbab', 'crown'])            
        elif opt in ('-x', '--sample'):
            sample = str(arg)
            sample = eval(sample) # for image data, we use nSamples
        elif opt in ('-e', '--eps'):
            eps = float(arg)
            assert 0. < eps <= 1.
        elif opt in ('-s', '--sort'):
            sorting = True  
        elif opt in ('-S', '--min'):
            smallest = True                       
        elif opt in ('-X', '--xtype'):
            xtype = str(arg) 
            assert (xtype in ['abd', 'con'])
        elif opt in ('-v', '--verb'):
            verb += 1
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)
    
    return sample, xtype, smallest, eps, alg, ncpu, featD, oracle, sorting, xnum, verb, args    
#
#==============================================================================
def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options]  model')
    print('Options:')
    print('        -a, --alg=<str>            Algo to compute expl: {swift, linear, dicho} (default = swift)')    
    print('        -h, --help')
    print('        -c, --cpu=<int>            number of cpu/proc to launch in parallel-dicho/swift algo (default = 16)')
    print('        -e, --eps=<float>          epsilon-ball (default = 0.08)')
    print('        --enum=<int>               number of XPs to enumerate')
    print('        --FD=<float>               feature len (in %) to activate feature-disjunction: [0, 1] (default = 0.75)')    
    print('        -o, --oracle=<str>         robustness reasoner: {marabou, mnbab, crown} (default = marabou)')
    print('        -s, --sort                 Sorting feats/pixels using LIME expl, otherwise apply sensitivity order')
    print('        -S, --min                  Smallest/Minimum-size  AXp/CXp explanation')
    print('        -x, --sample=<csv>         Explain the prediction of the given data input')
    print('        -X, --xtype=<str>          Explanation type: {abd, con} (default = abd)')
    print('        -v, --verb                 Be verbose (show comments)')

    

#
#==============================================================================
if __name__ == '__main__':
    
    plot_expl = False
    dataname = 'MNIST'

    nSamples, xtype, sxp, epsilon, xalg, ncpu, FD, oracle, sorting, nxp, verb, files = parse_options() 
    xconfig = XConfig(alg=xalg, featD=FD, approx=True, sort=sorting, slv=oracle)
    
    if len(files) == 0:
        print('.pth/.onnx/.nnet file is missing!')
        exit()     
    nn_fname = files[0]
    
    for name in ['MNIST', 'CIFAR10', 'GTSRB', 'TinyTaxiNet', 'SEMEION']:
        if name.lower() in nn_fname.lower():
            dataname = name 
            print(dataname)  
#     torch.no_grad()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     np.set_printoptions(precision=1, suppress=True)

    #cls = DNN(nn_fname) 
    xpl = RobXplainer(nn_fname, xconfig, ncpu=ncpu) 
    
    if len(files) == 1:
        import torch    
        import torchvision
        import torchvision.transforms as transforms 
        
        # load dataset
        BATCH_SIZE = 200
        if dataname == 'MNIST':
            # eps =  5%, 3%
            # LB tensor(0.)  UB tensor(1.)
            dataset = torchvision.datasets.MNIST(root=os.getcwd()+'/scratch/data/MNIST', train=False,
                                            transform=transforms.Compose([transforms.ToTensor(),]))
        elif dataname == 'CIFAR10':
            # eps = ??%
            # LB tensor(0.)  UB tensor(1.)
            dataset = torchvision.datasets.CIFAR10(root=os.getcwd()+'/scratch/data/CIFAR10', train=False, download=False, 
                                            transform=transforms.Compose([transforms.ToTensor(),]))
        elif dataname == 'SEMEION':
            # eps =  5%, 3%
            # LB tensor(0.)  UB tensor(1.)
            dataset = torchvision.datasets.SEMEION(root=os.getcwd()+'/scratch/data/SEMEION', download=False, 
                                            transform=transforms.Compose([transforms.Resize([16,16]), \
                                                                          transforms.ToTensor(),]))
        elif dataname == 'GTSRB':
            # eps =  0.5%, 3%
            # min tensor(0.)  max tensor(1.)
            dataset = torchvision.datasets.GTSRB(root=os.getcwd()+'/scratch/data/GTSRB',split="test", download=False, \
                                             transform=transforms.Compose([transforms.Resize([32,32]), \
                                                                           transforms.ToTensor(),]))
            top10= [1, 2, 13, 12, 38, 10, 4, 5, 25, 9]
            indices = [idx for idx, inst in enumerate(dataset) if inst[1] in top10]
            #testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices),
            #                             batch_size=BATCH_SIZE, shuffle=True)
        else:
            assert False, 'Dataset not available'
        
        testloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        #print(dataset.data.size(), dataset.targets.size())

        if dataname == 'GTSRB':
            testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices),
                                         batch_size=BATCH_SIZE, shuffle=True)            
          
        batch, y = next(iter(testloader))
        #print(batch.size())
        #print('LB', torch.min(batch),' UB', torch.max(batch))
        # labels = torch.argmax(cls(batch.to(device)), dim=1)
        inputs = batch.data.cpu().numpy()
        labels = np.argmax(xpl.dnn(inputs), axis=1) 
        #print(labels.unique())
        print(labels)
        with open(os.getcwd()+f'/scratch/data/{dataname}/inst.npy', 'wb') as fp:
            np.save(fp, inputs)
            #np.save(fp, labels)
        exit(0)                 
    else:
        with open(files[1], 'rb') as fp:
            inputs = np.load(fp)
            #labels = np.load(fp)
            labels = np.argmax(xpl.dnn(inputs), axis=1)
            
    results = []
    times = []
    lengths = []
    
    #tested = 0
    tested = nSamples - 1 # for experiments
    
    for i in range(inputs.shape[0]):
        if tested+i >= nSamples:
            break
        
        if nxp > 1:
            j = 0
            for expl in xpl.enumerate((inputs[i], labels[i]), eps=epsilon, xnum=nxp):
                j += 1
                print('len(xp):', len(expl))
            
        else:
            expl =  xpl.explain((inputs[i], labels[i]), eps=epsilon, xtype=xtype, optim=sxp)

        # times.append(xpl.time)
        # lengths.append(len(expl))
        # results.append({})
        # results[-1]['expl'] = expl
        # results[-1]['len'] = len(expl)
        # results[-1]['time'] = f'{xpl.time:.2f}'

        if plot_expl:
            #img = np.transpose(np.squeeze(inputs[i]))
            img = np.transpose(inputs[i])
            mask = np.zeros(xpl.assums.shape).astype(bool)
            mask[expl] = True
            if nxp > 1:
                mask = (xpl.ffa * 100).astype(np.int32)
            plot_shape = img.shape[0:2] if dataname == "MNIST" else img.shape
            root = str(Path(__file__).resolve().parent/'results/plots/')
            save_figure(image=label2rgb(mask.reshape(img.shape[0:2]),
                                        img.reshape(plot_shape),
                                        colors=[[0, 0.6, 0.5]],
                                        bg_label=0,
                                        saturation=1),
                        path=root+f"/{dataname.lower()}-expl#{tested+i+1}.png" )        
            save_figure(image=img,
                        path=root+f"/{dataname.lower()}-img#{tested+i+1}.png",
                        cmap="gray" if dataname == 'MNIST' else None)
    
    # delete robustness oracles and xplainer object        
    del xpl
            
# ../../../robxpl.py -x 1 --eps 0.03 --alg linear -o mnbab ../../../../benchmarks/onnx/mnist/MNIST_dense.onnx ~/inst.npy