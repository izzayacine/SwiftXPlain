
import resource
#import collections
from tqdm import tqdm, trange
from timeit import default_timer as timer

import torch.multiprocessing as mp

from rx_utils import *
from .xptype import *

class CXp(Explanation):

    def __init__(self, xpl, verb=1):
        super().__init__(xpl)
        self.verbose = verb

    def dicho(self, hypos):
        """
            dichotomic search alg to compute 1 cxp/mcs
        """
        core = []
        #hypos = [hypos[i] for i in self.x.order]

        self.x.ncalls = 0
        times = []
        while len(hypos): # not self.x.slv.has_aex(fixed=core)
            lb, ub = 0, len(hypos)-1
            while lb < ub:
                mid = (lb + ub) // 2
                t0 = timer()
                to_test = core+hypos[:mid+1]
                to_test = [h for h in self.x.assums if (h not in to_test)]
                #aexok = self.x.slv.has_aex(to_test)
                #===============================
                aexok = self.x.has_aex(to_test)
                # aexok = self.x.has_aex3(to_test)
                #==========================  
                times.append(timer()-t0)               
                self.x.ncalls += 1
                if not aexok:
                    lb = mid+1
                else:    
                    ub = mid
                
                if ub == lb:
                    core.append(hypos[ub])
                    hypos = hypos[:ub]
                    break
            #        
        if self.verbose > 0:
            #print(np.array([1 if (i in core) else 0 for i in self.x.order]))
            print('avg time:',f'{sum(times)/len(times):.3f}')
            print('max time:',f'{max(times)}')
            print('min time:', f'{min(times)}')        
        return core   

    
    def linear(self, hypos):
        """
            deletion (linear search) algo to compute 1 cxp/mcs
        """
        core = []
        approx = [i for i in self.x.assums if (i not in hypos)]
        #hypos = [hypos[i] for i in self.x.order]
        
        times = []
        self.x.ncalls = len(hypos)
        a, b = len(approx),len(self.x.assums)
        for j,_ in enumerate(trange(a,b,initial=a, total=b)):
            t0 = timer()
            to_test = [h for h in hypos[:j] if (h not in core)] + hypos[j+1:] + approx
            # aexok = self.x.slv.has_aex(fixed=to_test)
            #===============================
            aexok = self.x.has_aex(to_test) # incremental calls
            # aexok = self.x.has_aex3(to_test)
            #==========================    
            if not aexok:
                core.append(hypos[j])   
            times.append(timer()-t0)    
        if self.verbose > 0:
            #print(np.array([1 if (i in core) else 0 for i in self.x.order]))
            #print([1 if (i in core) else 0 for i in self.x.order])
            print('avg time:',f'{sum(times)/len(times):.3f}')
            print('max time:',f'{max(times)}')
            print('min time:', f'{min(times)}')    
        return core 

    
    def swift(self, hypos):
        """
            Parallel dichotomic search to compute 1 CXp/MCS
            return core ⊆ approx
        """
        #self.fixed = [i for i in hypos if (i not in self.x.order)]           
        #hypos = [hypos[i] for i in self.x.order]
        
        # ordered hypos f_i
        self.fixed = [i for i in self.x.assums if (i not in hypos)]
        self.approx = hypos.copy() # weak cxp
        core = []

        self.x.ncalls = 0
        self.x.fdCalls = 0
        self.x.fd_success = 0
        otimes = []

        pbar = tqdm(total=len(hypos))
        pbar.set_description("Processing")         
        
        while (len(hypos)):
            step_pbar = len(hypos)
            # feature disjunction (parallel CLD)
            if (len(hypos) <= (1. - self.x.opts.featD)*len(self.x.assums)):
                t0 = timer()
                k = min(self.x.mxp2, len(hypos))
                n = len(hypos)
                hypos, core = self.FeatDisjunct(hypos[:-k], core, hypos[-k:])
                
                self.x.ncalls += 1
                self.x.fdCalls += 1
                pbar.update(step_pbar - len(hypos))
                otimes.append(timer()- t0)
                continue        
                        
            l, u = 0, len(hypos)
            while l+1 < u:
                njob = min(self.x.mx_proc, u-l)
                assert (njob >= 2)
                step = (u-l) / njob
                chnks = [l+int(i*step) for i in range(1,njob)] 
                # print('>>', chnks)
                # print(len(hypos), len(chnks), njob, step)

                t0 = timer()
                # termincond = mp.Condition()
                # res = [mp.Value('i', -1) for _ in range(len(chnks))]
                # threads = []
                # for i,j in enumerate(chnks):
                #     to_test = [h for h in self.approx[j:] if (h not in core)] + self.fixed
                #     threads.append(mp.Process(target=run_solver3, args=(self.x.vnn_slv, self.x.inst, self.x.eps, \
                #                                             to_test, res[i], termincond))) 
                                           
                # for thread in threads:
                #     thread.start()

                # # wait until geting AEx (i.e. True), or
                # # until all the chunks have completed                    
                # with termincond:
                #     termincond.wait_for(lambda:advx_ok(res), timeout=TIME_LIMIT)
                #     i = next((len(res)-i-1 for i,x in enumerate(res[::-1]) if x.value == 0), 0) # last ¬AEx
                #     j = next((j for j in range(len(res)) if res[j].value == 1), len(res)) # first AEx
                #     res2 = [i.value for i in res]
                #     for proc in threads[:i]+threads[j:]:
                #         #proc.terminate()
                #         proc.kill()                
                # for thread in threads:
                #     thread.join(timeout=0)
                
                res = [-1 for _ in range(len(chnks))] # ∈ {-1,0,1}
                for i,j in enumerate(chnks):
                    to_test = [h for h in self.approx[j:] if (h not in core)] + self.fixed
                    res[i] = self.x.has_aex(to_test, i)
                i = next((len(res)-i-1 for i,x in enumerate(res[::-1]) if x == 0), 0) # last ¬AEx
                j = next((j for j in range(len(res)) if res[j] == 1), len(res)) # first AEx

                self.x.ncalls += 1
                otimes.append(timer()- t0)
                # print(res2)
                # print((l,u),'>>',end='')
                
                # update upper & lower bounds
                if j<len(res):
                    u = chnks[j]
                if j>0:        
                    l = chnks[j-1]
                
                # print((l,u))

            if u == 1: 
                to_test = [h for h in self.approx if (h not in core)] + self.fixed
                aexok = self.x.has_aex(to_test)
                # aexok = self.x.has_aex3(to_test)
                if aexok:
                    return core
            
            core.append(hypos[u-1])
            hypos = hypos[:u-1]
            
            pbar.update(step_pbar - len(hypos))
            
            #assert self.x.slv.has_aex([h for h in self.x.assums if (h not in core+hypos)])

        pbar.close()
        if self.verbose > 0:
            #print(np.array([1 if (i in core) else 0 for i in self.x.order]))
            #print([1 if (i in core) else 0 for i in self.x.order])
            print('avg time:',sum(otimes)/len(otimes))
            print('max time:', max(otimes))
            print('min time:', min(otimes))
            print('FD calls:', self.x.fdCalls)
            print('FD success:', self.x.fd_success)

        return core        


    def FeatDisjunct(self, hypos, core, fts): 
        """
            feature-D
        """
        
        to_tests = []
        for i in range(len(fts)):
            free = core+fts[:i]+fts[i+1:]
            to_tests.append([h for h in self.approx[len(hypos):] if (h not in free)] + self.fixed)
        
        # res = [mp.Value('i', -1) for _ in range(len(fts))]
        # threads = [mp.Process(target=run_solver3, args=(self.x.vnn_slv, self.x.inst, self.x.eps, test, res[i])) 
        #             for i, test in enumerate(to_tests)]        
        # for thread in threads:
        #     thread.start()
        # for thread in threads:
        #     thread.join(timeout=TIME_LIMIT)
        # res = [x.value for x in res]

        res = [-1 for _ in range(len(fts))]
        for i, test in enumerate(to_tests):
            res[i] = self.x.has_aex(test, gpu_id=i)        

        if all([(x == 0) for x in res]):
            core += fts
            if self.verbose > 1:
                print('#f_j added',len(fts))
            self.x.fd_success += 1    
            return hypos, core
            
        # atleast1 ¬AEx
        j = next((i for i, aex in enumerate(res) if (aex == 1)), None) 
        if j is None:
            warnings.warn("proc timeout, final output is a WAXp")
            core+= fts # weak AXp
            return hypos, core

        del fts[j]

        return (hypos+fts, core)        