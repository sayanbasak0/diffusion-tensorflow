# coding: utf-8
# author: Sayan Basak
'''
Implementing the diffusion equation simulator on a N-D lattice
'''

import os
import numpy as np
import time
import json

class Simulator:
    def __init__(self, n=(100,), dL=1, bc=0, t=0):
        '''
        Initialize the simulator with given parameters
        Args:
            n (list(int) / tuple(int)):
                Shape of the N-D lattice grid
            dL (float):
                lattice spacing
            bc (int / string / list(int / string) ):
                Boundary Condition
                0 -> 'closed' particle conserving| 1 -> 'open'
                int/string, all boundaries are same 
                or list of size 2*N 
                or list of list, where the former has size N, and the nested one has size 2
                or list of size N, both boundary on that axis has the same boundary condition
            t (float):
                Simulation time (not elapsed time to run)
        Returns:
            Simulator instance
        '''
        self.dL = dL
        self.n = tuple(n)
        self.t = t
        self.boundary_condition = [['closed','closed'] for ni in self.n]
        self.bc = [[0,0] for ni in self.n]
        self.exBC = [[1,ni+1] for ni in self.n]
        self.nBC = tuple([ni+2 for ni in self.n])
        self.n_i = [self.n[:i]+self.n[i+1:] for i in range(len(self.n))]
        if type(bc)==list and len(bc)==2*len(self.n):
            for i in range(len(bc)):
                if (type(bc[2*i])==int and bc[2*i]==1) or (type(bc[2*i])==str and bc[2*i]=='open'):
                    self.bc[i][0] = 1
                    self.boundary_condition[i][0] = 'open'
                if (type(bc[2*i+1])==int and bc[2*i+1]==1) or (type(bc[2*i+1])==str and bc[2*i+1]=='open'):
                    self.bc[i][1] = 1
                    self.boundary_condition[i][1] = 'open'
                if (bc[2*i] not in [0,1,'open','closed']) or (bc[2*i+1] not in [0,1,'open','closed']):
                    print("Invalid boundary condition!")
                    return
        elif type(bc)==list and len(bc)==len(self.n):
            for i in range(len(bc)):
                if type(bc[i])==list and len(bc[i])==2:
                    if (type(bc[i][0])==int and bc[i][0]==1) or (type(bc[i][0])==str and bc[i][0]=='open'):
                        self.bc[i][0] = 1
                        self.boundary_condition[i][0] = 'open'
                    if (type(bc[i][1])==int and bc[i][1]==1) or (type(bc[i][1])==str and bc[i][1]=='open'):
                        self.bc[i][1] = 1
                        self.boundary_condition[i][1] = 'open'
                    if (bc[i][0] not in [0,1,'open','closed']) or (bc[i][1] not in [0,1,'open','closed']):
                        print("Invalid boundary condition!")
                        return
                elif (type(bc[i])==int and bc[i]==1) or (type(bc[i])==str and bc[i]=='open'):
                    self.bc[i][0] = 1
                    self.boundary_condition[i][0] = 'open'
                    self.bc[i][1] = 1
                    self.boundary_condition[i][1] = 'open'
                elif (type(bc[i])==int and bc[i]==0) or (type(bc[i])==str and bc[i]=='closed'):
                    pass
                else:
                    print("Invalid boundary condition!")
                    return
        elif (type(bc)==int and bc==1) or (type(bc)==str and bc=='open'):
            for i in range(len(self.n)):
                self.bc[i][0] = 1
                self.boundary_condition[i][0] = 'open'
                self.bc[i][1] = 1
                self.boundary_condition[i][1] = 'open'
        elif (type(bc)==int and bc!=0) or (type(bc)==str and bc!='closed'):
            print("Invalid boundary condition!")
            return
        self.bound = ','.join([f"{si}:{ei}" for si,ei in self.exBC])
        self.gbound = [None for i in range(2*len(n))]
        for i in range(len(n)):
            self.gbound[2*i+0] = ','.join([f"{si}:{ei}" if ii!=i else f"{si-1}:{ei-1}" for ii,[si,ei] in enumerate(self.exBC)])
            self.gbound[2*i+1] = ','.join([f"{si}:{ei}" if ii!=i else f"{si+1}:{ei+1}" for ii,[si,ei] in enumerate(self.exBC)])

        self.bbound = [None for i in range(2*len(n))]
        self.cbound = [None for i in range(2*len(n))]
        for i in range(len(n)):
            self.bbound[2*i+0] = ','.join([f"{si}:{ei}" if ii!=i else '0' for ii,[si,ei] in enumerate(self.exBC)])
            self.cbound[2*i+0] = ','.join([f"{si}:{ei}" if ii!=i else '1' for ii,[si,ei] in enumerate(self.exBC)])
            
            self.bbound[2*i+1] = ','.join([f"{si}:{ei}" if ii!=i else f"{ei}" for ii,[si,ei] in enumerate(self.exBC)])
            self.cbound[2*i+1] = ','.join([f"{si}:{ei}" if ii!=i else f"{ei-1}" for ii,[si,ei] in enumerate(self.exBC)])
            
        self.f = np.zeros(self.nBC)
        self.fgrad = np.zeros((2*len(self.n),)+self.n)
        self.D = np.zeros(self.nBC)
        self.Davg = np.zeros((2*len(self.n),)+self.n)
        self.B = np.zeros(self.nBC)
        self.Bavg = np.zeros((2*len(self.n),)+self.n)
        self.Bmodify = False
        
    def Xgrads(self, X, Xgrad):
        '''
        Calculate gradient with nearest neighbor of X variable and stores in Xgrad variable
        '''
        for i in range(2*len(self.n)):
            exec(f"self.{Xgrad}[{i}] = self.{X}[{self.gbound[i]}]-self.{X}[{self.bound}]")
    def Xavgs(self, X, Xavg):
        '''
        Calculate nearest neighbor average of X variable and stores in Xavg variable
        '''
        for i in range(2*len(self.n)):
            exec(f"self.{Xavg}[{i}] = self.{X}[{self.gbound[i]}]+self.{X}[{self.bound}]")
    
    def fgrads(self):
        '''
        Calculate gradient with nearest neighbor of f and stores in fgrad
        '''
        for i in range(2*len(self.n)):
            exec_string = f"self.fgrad[{i}] = (self.f[{self.gbound[i]}]-self.f[{self.bound}])/self.dL"
            exec(exec_string)
    
    def Davgs(self):
        '''
        Calculate nearest neighbor average of D and stores in Davg
        '''
        for i in range(2*len(self.n)):
            exec_string = f"self.Davg[{i}] = self.D[{self.gbound[i]}]+self.D[{self.bound}]"
            exec(exec_string)
    def Bavgs(self):
        '''
        Calculate nearest neighbor average of B and stores in Bavg
        '''
        for i in range(2*len(self.n)):
            exec_string = f"self.Bavg[{i}] = self.B[{self.gbound[i]}]+self.B[{self.bound}]"
            exec(exec_string)
    
    def set_f_with_b(self, f_with_b):
        '''
        Set/initialize f including the boundary values 
        Args:
            f_with_b (float/array(float)):
                Sets f to value(s) provided
                Array shape includes boundary, so, it must be of shape n+2
        Returns:
            None
        '''
        self.f *= 0
        self.f += np.broadcast_to(np.array(f_with_b), shape=self.nBC)
    def set_f(self, f):
        '''
        Set/initialize f excluding the boundary values
        Args:
            f (float / array(float)):
                Sets f to value(s) provided
                Array shape excludes boundary, so, it must be of shape n
                Boundary values are set to the values at the edges for 'closed' boundary condition
        Returns:
            None
        '''
        exec_string = f"self.f[{self.bound}] = np.broadcast_to(np.array(f), shape={self.n})"
        exec(exec_string)
        for i in range(len(self.bc)):
            for j in range(2):
                if self.bc[i][j]==0:
                    exec_string = f"self.f[{self.bbound[2*i+j]}] = self.f[{self.cbound[2*i+j]}]"
                    exec(exec_string)
    def set_f_at_b(self, f_at_b):
        '''
        Set/initialize f at the boundary values
        Args:
            f_at_b (list(float) / list(array(float))):
                Sets f at boundary to value(s) provided
                Each element of list must be float of array of appropriate shape
                e.g.: in 1D f_at_b[0] is first boundary element
                            f_at_b[1] is last boundary element
                e.g.: in 2D f_at_b[0] is first boundary elements of y-axis
                            f_at_b[1] is last boundary elements of y-axis
                e.g.: in 3D f_at_b[0] is first boundary elements of y-z plane
                            f_at_b[1] is last boundary elements of y-z plane
                the boundary value can be modified for open boundary condition only
                list element(s) None skips modifying a particular boundary
        Returns:
            None
        '''
        if type(f_at_b)==list and len(f_at_b)==2*len(self.n):
            for i in range(len(self.bc)):
                for j in range(2):
                    if self.bc[i][j]==1:
                        if type(f_at_b[2*i+j])!=type(None):
                            exec_string = f"self.f[{self.bbound[2*i+j]}] = np.broadcast_to(np.array(f_at_b[{2*i+j}]), shape={self.n_i[i]})"
                            exec(exec_string)
        else:
            print(f"f_at_b must be a list of size 2*{len(self.n)}!")
            return
    
    def set_D_with_b(self, D_with_b):
        '''
        Set/initialize D including the boundary values 
        Args:
            D_with_b (float / array(float)):
                Sets D to value(s) provided
                Array shape includes boundary, so, it must be of shape n+2
        Returns:
            None
        '''
        self.D *= 0
        self.D += np.broadcast_to(np.array(D_with_b), shape=self.nBC)
        self.Davgs()
    def set_D(self, D):
        '''
        Set/initialize D excluding the boundary values
        Args:
            D (float / array(float)):
                Sets D to value(s) provided
                Array shape excludes boundary, so, it must be of shape n
                Boundary values are set to the values at the edges
        Returns:
            None
        '''
        exec_string = f"self.D[{self.bound}] = np.broadcast_to(np.array(D), shape={self.n})"
        exec(exec_string)
        for i in range(len(self.bc)):
            for j in range(2):
                exec_string = f"self.D[{self.bbound[2*i+j]}] = self.D[{self.cbound[2*i+j]}]"
                exec(exec_string)
        # self.Xavgs(X='D',Xavg='Davg')
        self.Davgs()
    
    def set_B_with_b(self, B_with_b):
        '''
        Set/initialize B including the boundary values 
        Args:
            B_with_b (float/array(float)):
                Sets B to value(s) provided
                Array shape includes boundary, so, it must be of shape n+2
        Returns:
            None
        '''
        self.B *= 0
        self.B += np.broadcast_to(np.array(B_with_b), shape=self.nBC)
        self.Bavgs()
    def set_B(self, B):
        '''
        Set/initialize B excluding the boundary values
        Args:
            B (float / array(float)):
                Sets B to value(s) provided
                Array shape excludes boundary, so, it must be of shape n
                Boundary values are set to the values at the edgess
        Returns:
            None
        '''
        exec_string = f"self.B[{self.bound}] = np.broadcast_to(np.array(B), shape={self.n})"
        exec(exec_string)
        for i in range(len(self.bc)):
            for j in range(2):
                exec_string = f"self.B[{self.bbound[2*i+j]}] = self.B[{self.cbound[2*i+j]}]"
                exec(exec_string)
        # self.Xavgs(X='B',Xavg='Bavg')
        self.Bavgs()
        if (np.max(self.B)>0):
            self.Bmodify = True
        else:
            self.Bmodify = False
    
    def initialize(self, f=None, f_at_b=None, D=None, B=None):
        '''
        Set/initialize f_with_b or (f and/or f_at_b)
        and/or D excluding the boundary values
        and/or B excluding the boundary values
        Args:
            f (float / array(float)):
                See `set_f()`
                Use None to skip
            f_at_b (list(float) / list(array(float))):
                See `set_f_at_b()`
                Use None to skip
            D (float / array(float)):
                See `set_D()`
                Use None to skip
            B (float / array(float)):
                See `set_B()`
                Use None to skip
        Returns:
            None
        '''
        if type(f) != type(None):
            self.set_f(f)
        if type(f_at_b) != type(None):
            self.set_f_at_b(f_at_b)
        if type(D) != type(None):
            self.set_D(D)
        if type(B) != type(None):
            self.set_B(B)
    
    def update_grad(self):
        '''
        Calculates d(f)/dt according to modified diffusion equation and stores it in df_dt
        Returns:
            None
        '''
        self.fgrads()
        if self.Bmodify:
            self.fgrad[np.abs(self.fgrad)<self.Bavg] = 0
            self.fgrad[self.fgrad>self.Bavg] -= self.Bavg[self.fgrad>self.Bavg]
            self.fgrad[self.fgrad<-self.Bavg] += self.Bavg[self.fgrad<-self.Bavg]
        self.fgrad *= self.Davg
        self.df_dt = np.sum(self.fgrad,axis=0)/self.dL
    
    def update_f(self, ddt):
        '''
        Calculates f(t+ddt) using f(t), df_dt and ddt
        Args:
            ddt (float):
                small change in time to update f(t+ddt) = f(t) + (df_dt)*ddt
        Returns:
            None
        '''
        self.update_grad()
        exec_string = f"self.f[{self.bound}] += self.df_dt*ddt"
        exec(exec_string)
        for i in range(len(self.bc)):
            for j in range(2):
                if self.bc[i][j]==0:
                    exec_string = f"self.f[{self.bbound[2*i+j]}] = self.f[{self.cbound[2*i+j]}]"
                    exec(exec_string)
    
    def evolve(self, Dt=1, dt=0.1, plot_realtime_interval=0, plot_dt_steps=0, qdata=None):
        '''
        Evolves f for Dt duration using smaller time steps.
        Args:
            Dt (float):
                Duration of evolution
            dt (float):
                small change in time, if this is too large, subdivide to avoid instability
            plot_realtime_interval (float):
                real time interval in seconds to send data through multiprocessing Queue 
                this has higher priority than plot_dt_step
            plot_dt_step (int):
                evolution steps of dt after which data is sent data through multiprocessing Queue
            qdata (multiprocessing.Queue):
                a multiprocess queue to send data to be used by another process.
        Returns:
            None
        '''
        ddt = dt
        subt = int(np.ceil(2*2*len(self.n)*np.max(self.D/(self.dL**2))*ddt))
        subt = subt if subt>0 else 1
        ddt /= subt
        print(f"\rSimulating (ddt={ddt:.6f}) : t={self.t:.6f}-->", end='', flush=True)
        if plot_realtime_interval>0 and type(qdata)==type(mp.Queue()):
            qdata.put(["print",f"\rSimulating (ddt={ddt:.6f}) : t={self.t:.6f}-->"])
            tStart = time.time()
            for iDt in range(int(Dt/dt)):
                for idt in range(subt):
                    self.update_f(ddt=ddt)
                self.t += dt
                tNow = time.time()
                if tNow-tStart>plot_realtime_interval:
                    tStart = tNow
                    qdata.put(["plot",[self.t,self.f]])
            qdata.put(["plot",[self.t,self.f]])
            qdata.put(["print",f"{self.t:.6f} - Evolved!\n"])
            qdata.put(["Done",[self.t,self.f.copy()]])
        elif plot_dt_steps>0 and type(qdata)==type(mp.Queue()):
            qdata.put(["print",f"\rSimulating (ddt={ddt:.6f}) : t={self.t:.6f}-->"])
            plot_dt_steps = int(plot_dt_steps)
            iStep = 1
            for iDt in range(int(Dt/dt)):
                for idt in range(subt):
                    self.update_f(ddt=ddt)
                self.t += dt
                if iStep==plot_dt_steps:
                    iStep = 0
                    qdata.put(["plot",[self.t,self.f]])
                iStep += 1
            qdata.put(["plot",[self.t,self.f.copy()]])
            qdata.put(["print",f"{self.t:.6f} - Evolved!\n"])
            qdata.put(["Done",[self.t,self.f]])
        else:
            for iDt in range(int(Dt/dt)):
                for idt in range(subt):
                    self.update_f(ddt=ddt)
                self.t += dt
        print(f"{self.t:.6f} - Evolved!")
        # return self.f
    
    def steadyState(self, precision=0.0001, plot_realtime_interval=0, plot_dt_steps=0, qdata=None):
        '''
        Evolves f until change is lower than required precision in Steady State.
        Args:
            precision (precision):
                minimum change in f requiredto continue evolution
                maximum possible change in time(ddt) chosen to avoid instabilities
            plot_realtime_interval (float):
                real time interval in seconds to send data through multiprocessing Queue 
                this has higher priority than plot_dt_step
            plot_dt_step (int):
                evolution steps of ddt after which data is sent data through multiprocessing Queue
            qdata (multiprocessing.Queue):
                a multiprocess queue to send data to be used by another process.
        Returns:
            None
        '''
        subt = (2*2*len(self.n)*np.max(self.D/(self.dL**2)))
        ddt = 1/subt if subt>0 else 1
        print(f"\rSimulating (ddt={ddt:.6f}): t={self.t:.6f}-->",end='',flush=True)
        check = True
        if plot_realtime_interval>0 and type(qdata)==type(mp.Queue()):
            qdata.put(["print",f"\rSimulating (ddt={ddt:.6f}) : t={self.t:.6f}-->"])
            self.f_plot = self.f.copy()
            tStart = time.time()
            while check:
                self.update_f(ddt=ddt)
                self.t += ddt
                check = (precision<=np.max(np.abs(self.df_dt*ddt)))
                tNow = time.time()
                if tNow-tStart>plot_realtime_interval:
                    qdata.put(["plot",[self.t,self.f.copy()]])
                    tStart=tNow
            qdata.put(["plot",[self.t,self.f]])
            qdata.put(["print",f"{self.t:.6f} - Steady State reached!\n"])
            qdata.put(["Done",[self.t,self.f]])
        elif plot_dt_steps>0 and type(qdata)==type(mp.Queue()):
            qdata.put(["print",f"\rSimulating (ddt={ddt:.6f}) : t={self.t:.6f}-->"])
            plot_dt_steps = int(plot_dt_steps)
            iStep = 1
            while check:
                self.update_f(ddt=ddt)
                self.t += ddt
                check = (precision<=np.max(np.abs(self.df_dt*ddt)))
                if iStep==plot_dt_steps:
                    qdata.put(["plot",[self.t,self.f.copy()]])
                    iStep = 0
                iStep += 1
            qdata.put(["plot",[self.t,self.f]])
            qdata.put(["print",f"{self.t:.6f} - Steady State reached!\n"])
            qdata.put(["Done",[self.t,self.f]])
        else:
            while check:
                self.update_f(ddt=ddt)
                self.t += ddt
                check = (precision<=np.max(np.abs(self.df_dt*ddt)))
        print(f"{self.t:.6f} - Steady State reached!")
        # return self.f
    
    def save_checkpoint(self, filename):
        '''
        Saves current instance necessary (to recreate instance) varibles and arrays to files.
        Args:
            filename (string):
                filename with path and without extension
                2 files <filename>.json and <filename>.npy saved
        Returns:
            None
        '''
        folder = os.path.split(filename)[0]
        if len(folder)>0:
            os.makedirs(folder, exist_ok=True)
        data_vars = {}
        data_vars["n"] = self.n
        data_vars["bc"] = self.bc
        data_vars["dL"] = self.dL
        data_vars["t"] = self.t
        data_array = np.array([self.f,self.D,self.B])
        with open(f"{filename}.json", 'w') as f:
            json.dump(data_vars, f)
        np.save(f"{filename}.npy",data_array)
        print(f"Saved to: {filename}.json & {filename}.npy")
    def load_checkpoint(self, filename):
        '''
        Loads varibles and arrays from files and recreates saved instance.
        Args:
            filename (string):
                filename with path and without extension
                2 files <filename>.json and <filename>.npy required to load
        Returns:
            None
        '''
        with open(f"{filename}.json", 'r') as f:
            data_vars = json.load(f)
        self.__init__(n=data_vars["n"], dL=data_vars["dL"], bc=data_vars["bc"], t=data_vars["t"])
        data_array = np.load(f"{filename}.npy")
        self.set_f_with_b(f_with_b=data_array[0])
        self.set_D_with_b(D_with_b=data_array[1])
        self.set_B_with_b(B_with_b=data_array[2])
        print(f"Loaded: {filename}.json & {filename}.npy")
    
import traceback
import imageio
import skimage.transform
import matplotlib as mpl
import multiprocessing as mp
import matplotlib.pyplot as plt
import sys

class Animator:
    def __init__(self, simObj, savefile=None, fps=10, figsize=None, force_initial_size=True, fmin=None, fmax=None, display=None):
        '''
        Args:
            simObj (Simulator):  [required]
                Simulator instance
            savefile (string): 
                filename with .gif or .mp4 extension to save animations
            fps (int):
                frames per second for the animations
            figsize (float / tuple(float) / list(float)): 
                figure size to set the output to, default is 4
                float - square dimension
                list/tuple - must be size 2 for rectangular dimension
            force_initial_size (bool):
                Resizes figure before plotting, this helps keep the original dimensions of figure
                it will only take effect if saving to a file
                default True
                if False, does not force figsize, but results can look out of proportion in the saved animation
            display (bool): 
                True - to keep display on for live simulation 
                False - doesn't affect animation saving to file
            fmin (float):
                set minimum value of 'simObj.f' to show 
                1-D: set the minimum of y-axis
                2-D & 3-D: the colorscale minimum 
            fmax (float):
                set maximum value of 'simObj.f' to show 
                1-D: set the maximum of y-axis
                2-D & 3-D: the colorscale maximum 
        Returns:
            Animator instance
        '''
        if isinstance(simObj,Simulator):
            self.simObj = simObj
        else:
            raise Exception("simObj must be Simulator instance!")
        self.qdata = mp.Queue()
        if not figsize:
            figsize = [4,4]
        elif type(figsize) in [int,float]:
            figsize = [figsize,figsize]
        elif type(figsize) in [list,tuple]:
            if len(figsize)==1 and (type(figsize[0]) in [int,float]):
                figsize = [figsize,figsize]
            elif len(figsize)==2:
                if (type(figsize[0]) not in [int,float]) or (type(figsize[1]) not in [int,float]):
                    raise Exception("Invalid figsize this!")
            else:
                raise Exception("Invalid figsize that!")
        figsize = tuple(figsize)
        self.figsize = figsize
        self.force_initial_size = force_initial_size
        self.fig = plt.figure(figsize=figsize, dpi=128, constrained_layout=True)
        self.savegif = False
        if type(savefile)==str:
            if savefile.endswith('.gif') and len(savefile)>4:
                folder = os.path.split(savefile)[0]
                if len(folder)>0:
                    os.makedirs(folder, exist_ok=True)
                self.savegif = True
                self.savefile = savefile
                self.gifWriter = imageio.get_writer(self.savefile, mode='I', fps=fps)
            elif savefile.endswith('.mp4') and len(savefile)>4:
                folder = os.path.split(savefile)[0]
                if len(folder)>0:
                    os.makedirs(folder, exist_ok=True)
                self.savegif = True
                self.savefile = savefile
                self.gifWriter = imageio.get_writer(self.savefile, mode='I', fps=fps, macro_block_size=None)
            else:
                raise Exception("Can only save to *.gif or *.mp4 file! ")
        if type(display)==type(None):
            try:
                self.display = self.display
            except:
                self.display = True
        else:
            if display:
                self.display = True
                plt.ion()
            else:
                self.display = False
                plt.ioff()
        self.fmin = fmin
        self.fmax = fmax
        self.setup_plot()
        
    def __enter__(self):
        '''
        Same as __init__() to use `with` statement
        Returns:
            Animator instance
        '''
        return self
    def close(self):
        '''
        Deletes this instance
        calls __exit__() 
        Returns:
            None
        '''
        self.__exit__(None, None, None)
    def __exit__(self, exc_type, exc_value, tb):
        '''
        Deletes this instance
        Args:
            In case of exceptions occurs:
            exc_type (type(Exception)): 
                The class of the original traceback
            exc_value (Exception): 
                Exception 
            tb: 
                Exception traceback
        Returns:
            None
        '''
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        self.qdata.close()
        self.qdata.join_thread()
        if self.savegif:
            self.gifWriter.close()
            print(f"Animation saved to: {self.savefile}")
        # plt.close()

    def start_animation(self, simulate, kwargs={}, display=None):
        '''
        Starts the animation using the simulate method name provided
        Args:
            simulate (string):
                must be a name of simulation method/function of the Simulator instance
                curently - "evolve" or "steadyState"
            kwargs (dict):
                Keyword arguments passed to the corresponding simulation method
                default - 'plot_realtime_interval' = 0.1
                'plot_realtime_interval' supersedes 'plot_dt_step' if both are provided
            display (bool):
                Turn display on or off
        Returns:
            None
        '''
        if type(simulate)==str and simulate=="evolve":
            self.simulate = self.simObj.evolve
        elif type(simulate)==str and simulate=="steadyState":
            self.simulate = self.simObj.steadyState
        else:
            print(f"Not valid simulation function: {simulate}")
            return 
        
        if type(display)==type(None):
            try:
                self.display = self.display
            except:
                self.display = True
        else:
            if display:
                self.display = True
                plt.ion()
            else:
                self.display = False
                plt.ioff()
        
        if 'plot_realtime_interval' not in kwargs.keys():
            if 'plot_dt_steps' not in kwargs.keys():
                kwargs['plot_realtime_interval'] = 0.1
            
        kwargs['qdata'] = self.qdata
        self.kwargs = kwargs
        
        
        # Create new Process
        self.process1 = mp.Process(target=self.simulate, kwargs=kwargs)
        # Start new Process
        self.process1.start()
        
        self.update_plot()
        
        self.process1.join()
        
        # print("Done!")
        # input("Press enter to continue ... ")
    
    def setup_plot(self):
        '''
        Setup plot before animating
        Returns:
            None
        '''
        self.w,self.h = self.fig.canvas.get_width_height()
        
        if len(self.simObj.n)==1:
            self.setup1D([self.simObj.t,self.simObj.f])
        elif len(self.simObj.n)==2:
            self.setup2D([self.simObj.t,self.simObj.f])
        elif len(self.simObj.n)==3:
            self.setup3D([self.simObj.t,self.simObj.f])
        self.plot = lambda *args: self.plot1D(*args) \
            if len(self.simObj.n)==1 \
            else self.plot2D(*args) if len(self.simObj.n)==2 \
                else self.plot3D(*args) if len(self.simObj.n)==3 \
                    else self.plotND(*args)
    def update_plot(self):
        '''
        Update plot while simulating
        Returns:
            None
        '''
        while True:
            if not self.qdata.empty():
                data = self.qdata.get_nowait()
                if type(data[0])==str and data[0]=="print":
                    print(f"{data[1]}",end='',flush=True)
                    sys.stdout.flush()
                elif type(data[0])==str and data[0]=="Done":
                    self.simObj.t = data[1][0]
                    self.simObj.f = data[1][1]
                    break
                elif type(data[0])==str and data[0]=="plot":
                    self.plot(data[1])
        
    def setup1D(self,data):
        '''
        Setup 1-D simulation plot before animating
        Args:
            data (list(float,array)):
                't' and 'f' before simulation starts
        Returns:
            None
        '''
        t,f = data
        if not self.fmax:
            self.fmax = np.max(f)+0.000001
        if not self.fmin:
            self.fmin = np.min(f)-0.000001
        self.ax = self.fig.gca()
        self.ax.cla()
        self.line, = self.ax.plot(self.simObj.dL*(-0.5+np.arange(self.simObj.nBC[0])), f, marker='.')
        self.ax.set_xlim(-1,self.simObj.n[0]+1)
        self.ax.set_ylim(self.fmin,self.fmax)
        self.ax.set_xlabel(f"x")
        self.ax.set_ylabel(f"f(x,t)")
        self.ax.set_title(f"t={t:.6f}")
        self.fig.canvas.draw()
        if self.display:
            plt.pause(0.000001)
        if self.savegif:
            self.fig2frame(self.fig)
    def plot1D(self,data):
        '''
        Update 1-D simulation plot while animating
        Args:
            data (list(float,array)):
                't' and 'f' data from multiprocess Queue
        Returns:
            None
        '''
        t,f = data
        self.line.set_ydata(f)
        self.ax.set_title(f"t={t:.6f}")
        
        if self.savegif:
            self.fig2frame(self.fig)
        if self.display:
            self.fig.canvas.draw()
            plt.pause(0.000001)
    def setup2D(self,data):
        t,f = data
        if not self.fmax:
            self.fmax = np.max(f)+0.000001
        if not self.fmin:
            self.fmin = np.min(f)-0.000001
        self.ax = self.fig.gca()
        self.ax.cla()
        self.area = self.ax.imshow(f.T, vmin=self.fmin, vmax=self.fmax,  extent=[-1,self.simObj.dL*self.simObj.n[0]+1,-1,self.simObj.dL*self.simObj.n[1]+1])
        cbar = self.fig.colorbar(self.area, ax=self.ax)
        cbar.ax.set_ylabel('f(x,y,t)', rotation=90)
        self.ax.set_xlabel('x',)
        self.ax.set_ylabel('y',)
        self.ax.set_title(f"t={t:.6f}")
        self.fig.canvas.draw()
        if self.display:
            plt.pause(0.000001)
        if self.savegif:
            self.fig2frame(self.fig)
    def plot2D(self,data):
        '''
        Update 2-D simulation plot while animating
        Args:
            data (list(float,array)):
                't' and 'f' data from simulation
        Returns:
            None
        '''
        t,f = data
        self.area.set_data(f.T)
        self.ax.set_title(f"t={t:.6f}")
        if self.savegif:
            self.fig2frame(self.fig)
        if self.display:
            self.fig.canvas.draw()
            plt.pause(0.000001)
        
    def setup3D(self,data):
        '''
        Setup 1-D simulation plot before animating
        Args:
            data (list(float,array)):
                't' and 'f' before simulation starts
        Returns:
            None
        '''
        t,f = data
        opacity = 0.5 #1.0/np.min(self.simObj.n)
        if not self.fmax:
            self.fmax = np.max(f)+0.000001
        if not self.fmin:
            self.fmin = np.min(f)-0.000001
        self.ax = self.fig.gca(projection='3d')
        self.ax.cla()
        # make the panes transparent
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # # make the grid lines transparent
        self.ax.xaxis._axinfo["grid"]['color'] =  (0,0,0,0.2)
        self.ax.yaxis._axinfo["grid"]['color'] =  (0,0,0,0.2)
        self.ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,0.2)
        
        self.x,self.y,self.z = np.meshgrid(-0.5+np.arange(self.simObj.nBC[0]),-0.5+np.arange(self.simObj.nBC[1]),-0.5+np.arange(self.simObj.nBC[2]))
        self.volume = self.ax.scatter(self.x.flatten(),self.y.flatten(),self.z.flatten(), c=f.flatten(), alpha=0.5, edgecolor="none", marker='s', cmap='viridis_r', vmin=self.fmin, vmax=self.fmax)
        self.myCmap = self.volume.get_cmap()
        self.nCmap = lambda x: self.myCmap((x-self.fmin)/(self.fmax-self.fmin))
        cbar = self.fig.colorbar(self.volume, ax=self.ax)
        cbar.ax.set_ylabel('f(x,y,z,t)', rotation=90)
        self.ax.set_title(f"t={t:.6f}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim(-1,self.simObj.n[0]+1)
        self.ax.set_ylim(-1,self.simObj.n[1]+1)
        self.ax.set_zlim(-1,self.simObj.n[2]+1)
        self.fig.canvas.draw()
        if self.display:
            plt.pause(0.000001)
        if self.savegif:
            self.fig2frame(self.fig)
    def plot3D(self,data):
        '''
        Update 3-D simulation plot while animating
        Args:
            data (list(float,array)):
                't' and 'f' data from simulation
        Returns:
            None
        '''
        t,f = data
        self.volume._facecolor3d = self.nCmap(f.flatten())
        # self.volume.set_clim(vmin=self.fmin,vmax=self.fmax)
        self.ax.set_title(f"t={t:.6f}")
        if self.savegif:
            self.fig2frame(self.fig)
        if self.display:
            self.fig.canvas.draw()
            plt.pause(0.000001)
    def setupND(self,data):
        '''
        Not implemented
        Args:
            data (list(float,array)):
                't' and 'f' before simulation starts
        Returns:
            None
        '''
        # t,f = data
        pass
    def plotND(self,data):
        '''
        Not implemented
        Args:
            data (list(float,array)):
                't' and 'f' data from simulation 
        Returns:
            None
        '''
        # t,f = data
        pass
    def fig2frame(self, fig):
        """
        Convert a matplotlib figure to a 3D numpy array with RGBA channels and appends frame to gif file writer
        Args:
            fig (matplotlib.pyplot.figure):
                a matplotlib figure
        Returns:
            None
        """
        if self.force_initial_size:
            # Get the RGBA buffer from the figure
            self.fig.set_size_inches(self.figsize[0],self.figsize[1])
            self.fig.canvas.draw()
            buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
            buf.shape = ( self.h, self.w, 4 )
        
            # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
            buf = np.roll ( buf, 3, axis = 2 )
            
            self.gifWriter.append_data(buf)
        else:
            # Get the RGBA buffer from the figure
            buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
            w,h = self.fig.canvas.get_width_height()
            buf.shape = ( h, w, 4 )
        
            # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
            buf = np.roll ( buf, 3, axis = 2 )
            if (self.h!=h or self.w!=w):
                buf = skimage.transform.resize(buf, (self.h,self.w), preserve_range=True).astype('uint8')
            
            self.gifWriter.append_data(buf)
