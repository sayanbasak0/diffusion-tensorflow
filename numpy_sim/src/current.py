# coding: utf-8
# author: Sayan Basak
'''
Implementing the current diffusion equation simulator on a N-D lattice
'''

import os
import numpy as np
import time
import json
from .diffusion import Simulator, Animator

class Resistor(Simulator):
    def __init__(self, n=(100,), dL=1, bc=0, t=0):
        self = Simulator.__init__(self, n=n, dL=dL, bc=bc, t=t)
    
    def Davgs(self):
        '''
        Calculate nearest neighbor average of D and stores in Davg
        '''
        for i in range(2*len(self.n)):
            exec_string = f"self.Davg[{i}] = 1/(self.D[{self.gbound[i]}]+self.D[{self.bound}])"
            exec(exec_string)
    
    def set_R(self, R):
        '''
        Set/initialize R excluding the boundary values
        Args:
            R (float / array(float)):
                must be > 0
                Sets D to R value(s) provided
                Array shape excludes boundary, so, it must be of shape n
                Boundary values are set to 0
        Returns:
            None
        '''
        # self.set_D(R)
        exec_string = f"self.D[{self.bound}] = np.broadcast_to(np.array(R), shape={self.n})"
        exec(exec_string)
        for i in range(len(self.bc)):
            for j in range(2):
                exec_string = f"self.D[{self.bbound[2*i+j]}] = 0"
                exec(exec_string)
        # self.Xavgs(X='D',Xavg='Davg')
        self.Davgs()
    
    def initialize(self, f=None, f_at_b=None, R=None):
        '''
        Set/initialize f_with_b or (f and/or f_at_b)
        and/or D excluding the boundary values
        Args:
            f (float / array(float)):
                See `set_f()`
                Use None to skip
            f_at_b (list(float) / list(array(float))):
                See `set_f_at_b()`
                Use None to skip
            R (float / array(float)) > 0:
                See `set_R()`
                Use None to skip
        Returns:
            None
        '''
        if type(f) != type(None):
            self.set_f(f)
        if type(f_at_b) != type(None):
            self.set_f_at_b(f_at_b)
        if type(R) != type(None):
            self.set_R(R)
    