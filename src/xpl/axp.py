
import resource
#import collections
from tqdm import tqdm
from timeit import default_timer as timer

import torch.multiprocessing as mp
import warnings

from rx_utils import *
from .xptype import *

#
#===============================================================
class AXp(Explanation):

    def __init__(self, xpl, verb=1):
        super().__init__(xpl) 
        self.verbose = verb   

    def dicho(self, hypos):
        """
            dichotomic search alg to compute 1 mus
        """
        core = []
        #self.x.order = np.arange(len(self.x.assums))
        hypos = [hypos[i] for i in self.x.order]
        self.x.ncalls = 0
        times = []
        while self.x.slv.has_aex(fixed=core):
            #self.x.ncalls += 1
            lb, ub = 0, len(hypos) - 1
            while True:
                mid = (lb + ub) // 2
                t0 = timer()
                #aexok = self.x.slv.has_aex(core+hypos[:mid+1])
                #===============================
                aexok = self.x.has_aex(core+hypos[:mid+1])
                # aexok = self.x.has_aex3(core+hypos[:mid+1])
                #==========================  
                times.append(timer()-t0)               
                self.x.ncalls += 1
                if not aexok:
                    ub = mid
                else:    
                    lb = mid+1
                
                if ub == lb:
                    j = mid+1 if aexok else mid
                    hypos = hypos[:j+1]
                    core.append(hypos[j])
                    del hypos[-1]
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
            deletion-based (linear search) algo to compute an mus
        """
        core = []
        order = np.flip(self.x.order)
        hypos = [hypos[i] for i in order]
        
        times, times2 = [], []
        self.x.ncalls = len(hypos)
        import time
        for j in tqdm(range(len(hypos))):
            t0 = timer()
            to_test = core + hypos[j+1:]
            # aexok = self.x.slv.has_aex(fixed=to_test)
            #===============================
            aexok = self.x.has_aex(to_test)
            # aexok = self.x.has_aex3(to_test)    
            #==========================    
            if aexok:
                core.append(hypos[j])   
            times.append(timer()-t0)
        if self.verbose > 0:
            #print([1 if (i in core) else 0 for i in order])
            print('avg time:',f'{sum(times)/len(times):.3f}')
            print('max time:',f'{max(times)}')
            print('min time:', f'{min(times)}')
                
        return core
    
    
    # def swift(self, hypos):
    #     """
    #         Parallel dichotomic algo for computing abductive explantion (mus features)
    #         return core subset hypos
        
    #     """             
            
    #     #hypos = [hypos[i] for i in self.x.order]
    #     core = []

    #     lenW = len(hypos)
    #     prev = 0

    #     pbar = tqdm(total=len(hypos))
    #     pbar.set_description("Processing") 
    #     step = len(hypos)
    #     self.x.ncalls = 0
    #     self.x.fdCalls = 0
    #     self.x.fd_success = 0
    #     otimes = []
    #     singFeat = False

    #     self.x._threads = None

    #     while(len(hypos)):
    #         chnks = splitChnk(lenW, prev)
    #         lb = 0
    #         ub = len(chnks) - 1
    #         if not prev:
    #             singFeat = False
            
    #         if (len(hypos) < self.x.opts.featD*len(self.x.order)) and (not singFeat) and \
    #             (not (prev and (lenW > self.x.mx_proc))):
    #             # feature-disjunction procedure
    #             step = len(hypos)
    #             t0 = timer()

    #             k = self.x.mxp2 if len(hypos) > self.x.mxp2 else 0
    #             if prev:
    #                 k = lenW
    #             hypos, core = self.disjunct(hypos[:-k], core, hypos[-k:])
    #             prev = 0
    #             lenW = len(hypos)
                
    #             singFeat = ((step - len(hypos)) == 1)
    #             self.x.ncalls += 1
    #             self.x.fdCalls += 1
    #             pbar.set_description(f'ncalls: {self.x.ncalls}')
    #             pbar.update(step - len(hypos))
    #             step = len(hypos)
    #             otimes.append(timer()-t0)
                
    #             continue                


    #         while True:
    #             #assert (ub > lb)
    #             dicho = binsplit(lb, ub, self.x.mx_proc)

    #             #singFeat = False    
    #             t0 = timer()

    #             termincond = mp.Condition()
    #             res = [mp.Value('i', -1) for _ in range(len(dicho))] 
    #             # threads = [mp.Process(target=run_solver, args=
    #             #                       (self.x.oracles[i], hypos[:chnks[j]]+core, res[i], termincond)) 
    #             #            for i,j in enumerate(dicho)] # incremental mode
    #             threads = [mp.Process(target=run_solver3, args=(self.x.vnn_slv, self.x.inst, self.x.eps, \
    #                                                         hypos[:chnks[j]]+core, res[i], termincond)) 
    #                        for i,j in enumerate(dicho)]                
    #             for thread in threads:
    #                 thread.start()

    #             ##for thread in threads:
    #             ##    thread.join() # waits for all threads to complete their tasks

    #             # wait until geting ¬AEx (False), or
    #             # until all the chunks have completed                    
    #             with termincond:
    #                 termincond.wait_for(lambda:neg_advx_ok(res), timeout=TIME_LIMIT)
    #                 i = next((len(res)-i-1 for i,x in enumerate(res[::-1]) if x.value == 1), 0) # before AEx
    #                 j = next((j for j in range(len(res)) if not res[j].value), len(res)) # after ¬AEx
    #                 res2 = [i.value for i in res]
    #                 for proc in threads[:i]+threads[j:]:
    #                     #proc.terminate()
    #                     proc.kill()

    #             for thread in threads:
    #                 thread.join(timeout=0)
                
    #             self.x.ncalls += 1
    #             otimes.append(timer()- t0)
                
    #             res = res2
    #             #print([ p.name in  for p in threads])
    #             if self.verbose > 1:
    #                 print([i for i in res], 'lb=',lb, 'ub=',ub, dicho)

    #             # is there: AEx(W_i) + ¬AEx(W_i+1)?
    #             i = next((i for i, aex in enumerate(res) if not aex), len(res)-1) # default: w_r   
    #             aexok = next((True for aex in res if (aex == 1)), False)
    #             unsat = next((True for aex in res if (aex == 0)), False)
    #             assert (aexok or unsat)
    #             mid = dicho[i]
    #             # Either update UB or/and LB...
    #             if unsat:
    #                 ub = mid
    #                 hypos = hypos[:chnks[mid]]
    #             if aexok:
    #                 lb = dicho[len(res) - [x for x in res[::-1]].index(1) - 1] + 1 # last aex +1 

    #             #print('>> lb=',lb, 'ub=',ub, 'mid=',mid)
    #             assert not (aexok and (not unsat) and mid==ub), 'w_ub must always be 0'


    #             if not (i or mid or prev): # i=mid=prev=0
    #                 # test if ¬AEx(core)
    #                 aex = mp.Value('i', -1)
    #                 run_solver(self.x.slv, core, aex)
    #                 self.x.ncalls += 1
    #                 if not aex.value:
    #                     # assert (not self.x.slv.solve(assumptions=core))
    #                     hypos = []
    #                     break                        

    #             if (lb == ub):
    #                 if aexok and unsat:
    #                     assert mid>0
    #                     #hypos = hypos[:chnks[mid]]
    #                     r = chnks[mid] - chnks[mid-1]
    #                     if r>1:
    #                         # zoom-in
    #                         prev = chnks[mid-1]
    #                         lenW = r
    #                     else:
    #                         # culptit at chnks[mid]
    #                         core.append(hypos.pop())
    #                         prev = 0
    #                         lenW = len(hypos)
    #                         if self.verbose > 1:
    #                             print('culprint at:',chnks[mid], core[-1])
    #                     break    
    #                 elif aexok: # (...,1,ub)
    #                     if (chnks[ub] - chnks[mid] == 1):
    #                         # cluprit at w_ub
    #                         core.append(hypos[chnks[ub]-1])
    #                         hypos = hypos[:chnks[ub]-1]
    #                         prev = 0
    #                         lenW = len(hypos)
    #                         if self.verbose > 1:
    #                             print('culprint at:',chnks[mid], core[-1])
    #                     else:
    #                         # zoom-in w_ub
    #                         prev = chnks[mid]
    #                         lenW = chnks[ub] - chnks[mid]
    #                     break
    #                 else: # (lb=0,...)
    #                     assert unsat
    #                     #hypos = hypos[:chnks[mid]]
    #                     r = (chnks[mid] - chnks[mid-1]) if mid>0 else (chnks[mid] - prev)
    #                     if r>1:
    #                         # zoom in
    #                         if mid>0:
    #                             prev = chnks[mid-1]
    #                         lenW = r                                
    #                     else:
    #                         # culprit at w_mid
    #                         core.append(hypos.pop()) # hypos[:chnks[mid]-1] + del hypos[-1]
    #                         prev = 0
    #                         lenW = len(hypos)
    #                         if self.verbose > 1:
    #                             print('culprint at:',chnks[mid], core[-1])
    #                     break 

    #             elif (ub-lb == 1) and unsat: # (lb,0,...)
    #                 # test AEx in w_lb?
    #                 aex = mp.Value('i', -1)
    #                 run_solver(self.x.oracles[0], core+hypos[:chnks[lb]], aex)
    #                 self.x.ncalls += 1
    #                 #print(chnks[lb], chnks[mid], chnks[ub], chnks)
    #                 mid = lb if not aex.value else lb+1
    #                 hypos = hypos[:chnks[mid]] # mid=lb=ub-1 or mid=ub
    #                 r =  (chnks[mid] - chnks[mid-1]) if mid>0 else (chnks[mid] - prev)
    #                 if r>1:
    #                     # zoom in
    #                     if mid>0:
    #                         prev = chnks[mid-1]
    #                     lenW = r                                
    #                 else:
    #                     # culprit at w_mid
    #                     core.append(hypos.pop()) # hypos[chnks[mid]-1]
    #                     prev = 0
    #                     lenW = len(hypos)
    #                     if self.verbose > 1:
    #                         print('culprint at:',chnks[mid], core[-1])

    #                 break 
    #         pbar.update(step - len(hypos))
    #         pbar.set_description(f'ncalls: {self.x.ncalls}')
    #         step = len(hypos)        
    #     pbar.close() 

    #     if self.x._threads is not None:
    #         self.x.end_flag.set()
    #         for thread in self.x._threads:
    #             thread.kill()             
    #         for thread in self.x._threads:
    #             thread.join(timeout=0) 

    #     if self.verbose > 1:
    #         # verify core is an MUS
    #         assert not self.x.slv.has_aex(fixed=core)
    #         for i in range(len(core)):
    #             assert self.x.slv.has_aex(fixed=core[:i]+core[i+1:]), f'irrelevant lit: {core[i]}'
    #     if self.verbose > 0:
    #         #print(np.array([1 if (i in core) else 0 for i in self.x.order]))
    #         #print([1 if (i in core) else 0 for i in self.x.order])
    #         print('avg time:',sum(otimes)/len(otimes))
    #         print('max time:', max(otimes))
    #         print('min time:', min(otimes))
    #         print('FD calls:', self.x.fdCalls)
    #         print('FD success:', self.x.fd_success)
    #     return core 
    
    
    def swift(self, hypos):
        """
            Parallel dichotomic search to compute an AXp/MUS
            return core ⊆ hypos
        """
        #hypos = [hypos[i] for i in self.x.order]
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
                #     to_test = hypos[:j]+core
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
                print('chnks:',chnks)
                for i,j in enumerate(chnks):
                    to_test = hypos[:j]+core
                    res[i] = self.x.has_aex(to_test,i)
                print('>>> ',res)    
                # i = next((len(res)-i-1 for i,aex in enumerate(res[::-1]) if aex), 0) # last AEx
                j = next((j for j in range(len(res)) if (not res[j])), len(res)) # first ¬AEx

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
                # aexok = self.x.has_aex3(core)                
                aexok = self.x.has_aex(core)
                if not aexok:
                    return core
            
            core.append(hypos[u-1])
            hypos = hypos[:u-1]
            
            pbar.update(step_pbar - len(hypos))
            
            #assert self.x.has_aex([h for h in self.x.assums if (h not in core+hypos)])

        pbar.close()

        if self.verbose > 1:
            # verify core is an MUS
            assert not self.x.slv.has_aex(core)
            for i in range(len(core)):
                assert self.x.slv.has_aex(core[:i]+core[i+1:]), f'irrelevant lit: {core[i]}'
        if self.verbose > 0:
            #print(np.array([1 if (i in core) else 0 for i in self.x.order]))
            #print([1 if (i in core) else 0 for i in self.x.order])
            print('avg time:',sum(otimes)/len(otimes))
            print('max time:', max(otimes))
            print('min time:', min(otimes))
            print('FD calls:', self.x.fdCalls)
            print('FD success:', self.x.fd_success)

        return core        

#     def disjunct(self, hypos, core, fts): 
#         """
#             feature-D
#         """
#         if self.x._threads is None:
#             self.x.end_flag = mp.Event()
#             self.x.start_flags = [mp.Event() for _ in range(self.x.mx_proc)] 
#             self.x.completcond = mp.Condition()
#             self.x.results = [mp.Value('i', -1) for _ in range(self.x.mx_proc)]
#             self.x.arr_hypos = [mp.Array('i', [0]*len(self.x.order)) for _ in range(self.x.mx_proc)]  
#             self.x._threads = [mp.Process(target=run_solver2, args= (self.x.oracles[i], self.x.arr_hypos[i], self.x.results[i], \
#                                             self.x.completcond, self.x.start_flags[i], self.x.end_flag, i+1))  for i in range(self.x.mx_proc)]
#             for thread in self.x._threads:
#                     thread.start()            
# # #         with mp.pool.Pool() as pool:
# # #             to_tests = [fixhypos+core+hypos[:i]+hypos[i+1:] for i in range(len(hypos))]
# # #             # prepare_oracles(len(hypos) - 1)
# # #             #threads = pool.starmap(run_solver, zip(self.x.oracles, to_tests))
# # # #                     for thread in threads:
# # # #                         thread.wait(timeout=TIME_LIMIT)
# # #             pool.close()
# # #             pool.join()
                    
# #         with ProcessPoolExecutor() as ex:
# #             to_tests = [hypos+core+fts[:i]+fts[i+1:] for i in range(len(fts))]
# #             prepare_oracles(len(fts) - 1)
# #             #threads = ex.map(run_solver, self.x.oracles, to_tests)
# #             threads = [ex.submit(run_oracle, self.x.cnf, hs ) for hs in to_tests] 
# #             completed,_ = cf.wait(threads, timeout=TIME_LIMIT, return_when=cf.ALL_COMPLETED)
# #             assert len(completed), 'All process/oracles are timeout.'
                
# #         if ((len(completed) == len(fts)) and 
# #             all([thread.result() for thread in completed]) ):
# #             # fts must be in the AXp
# #             core += fts
# #             print('#f_j added',len(fts))
# #             return hypos, core
#         """
#         res = [mp.Value('i', -1) for _ in range(len(fts))]
#         to_tests = [hypos+core+fts[:i]+fts[i+1:] for i in range(len(fts))]
#         threads = [mp.Process(target=run_solver, args=(self.x.oracles[i], test, res[i])) 
#                     for i, test in enumerate(to_tests)]

#         for thread in threads:
#             thread.start()
#         for thread in threads:
#             thread.join(timeout=TIME_LIMIT)
#         """
#         def allresults(res):
#             return all([(x.value > -1) for x in res])

#         for i in range(self.x.mx_proc):
#             self.x.results[i].value = -1
#         to_tests = [hypos+core+fts[:i]+fts[i+1:] for i in range(len(fts))]
#         for i,test in enumerate(to_tests):
#             for t in test:
#                 self.x.arr_hypos[i][t] = 1    
#             self.x.start_flags[i].set()
#         with self.x.completcond:
#             self.x.completcond.wait_for(lambda:allresults(self.x.results[:len(fts)]), timeout=TIME_LIMIT) 
#             for s in self.x.start_flags:
#                 s.clear()
#             for i,test in enumerate(to_tests):
#                 for t in test:
#                     self.x.arr_hypos[i][t] = 0       
#         res = self.x.results[:len(fts)]
#         #print([x.value for x in res])
#         #===============================================
#         if all([(x.value == 1) for x in res]):
#             core += fts
#             if self.verbose > 1:
#                 print('#f_j added',len(fts))
#             return hypos, core
            
#         #assert any([not proc.result() for proc in completed]) # atleast1 ¬AEx
#         # print([x.value for x in res])
#         j = next(i for i, aex in enumerate(res) if not aex.value) 
#         del fts[j]

#         return (hypos+fts, core)       


    def FeatDisjunct(self, hypos, core, fts): 
        """
            feature-D
        """
        
        to_tests = [hypos+core+fts[:i]+fts[i+1:] for i in range(len(fts))]
        # res = [mp.Value('i', -1) for _ in range(len(fts))]
        # threads = [mp.Process(target=run_solver3, args=(self.x.vnn_slv, self.x.inst, self.x.eps, test, res[i])) 
        #             for i, test in enumerate(to_tests)]        
        # for thread in threads:
        #     thread.start()
        # for thread in threads:
        #     thread.join(timeout=TIME_LIMIT)
        print(len(to_tests))
        res = [-1 for _ in range(len(fts))]
        for i, test in enumerate(to_tests):
            res[i] = self.x.has_aex(test, gpu_id=i)

        if all([(x == 1) for x in res]):
            core += fts
            if self.verbose > 1:
                print('#f_j added',len(fts))
            self.x.fd_success += 1    
            return hypos, core
            
        #assert any([not proc.result() for proc in completed]) # atleast1 ¬AEx
        # print([x for x in res])
        j = next((i for i, aex in enumerate(res) if not aex), None) 
        if j is None:
            warnings.warn("proc timeout, final output is a WAXp")
            core+= fts # weak AXp
            return hypos, core

        del fts[j]

        return (hypos+fts, core)         
