from abc import ABC, abstractmethod
#from this import d
#from typing import *
import utility
from joblib import Parallel, delayed
import numpy as np
from liquid import Liquid
import pandas as pd
import copy
import multiprocessing as mp
from blackbox import BlackBox

class SimpleCase(BlackBox):
    def __init__(self, run_id, run_folder, lower, upper, settings):
        super().__init__()
        self.__run_id = run_id
        self.__run_folder = run_folder
        self.__settings = settings
        self.__lower = lower
        self.__upper = upper

    def setup(self):
        pass

    def rescale_design(self, design):
        c_design = copy.deepcopy(design)
        for j in range(0, len(c_design)):
            l = self.__lower[j]
            u = self.__upper[j]
            c_design[j] = utility.rescale_val(c_design[j], 0, 1, l, u)
        return c_design

    def obj_fun(self, design, with_metadata=False, infill_id=None):
        """ Objective function
        Args:
            design: A numpy series containing the design (1 row)
        Returns:
            error: the error as an np array
            response: additional details
        """
        eps = np.random.normal(0, 0.5, 1)[0]
        responses_dict = {}
        if with_metadata:
            # Index and infill_id at loc 0 and 1            
            design = design[2:len(design)]

        design = self.rescale_design(design)
        x = design[0]
        y = design[1]
        r = np.sqrt(x*x + y*y) + eps
        z = -1*np.sin(r)/r
        print(f"({x}, {y}) => {z}")
        return [z], responses_dict