from abc import ABC, abstractmethod
import utility
import numpy as np
import copy
from blackbox import BlackBox

class SimpleCase(BlackBox):
    def __init__(self, run_id, run_folder, lower, upper, settings):
        super().__init__()
        self.name = "SimpleCase"
        self.__run_id = run_id
        self.__run_folder = run_folder
        self.__settings = settings
        self._lower = lower
        self._upper = upper

    def setup(self, should_rerun_epsc_copy=True, should_refit_exp_data=True):
        pass

    def rescale_design_to_cpm(self, design):
        c_design = copy.deepcopy(design)

        for j in range(0, len(self._lower)):
            l = self._lower[j]
            u = self._upper[j]
            c_design[j] = utility.rescale_val(c_design[j], 0, 1, l, u)
        return c_design

    def obj_fun(self, design, is_initial_design_point, with_metadata=False, infill_id=None, rescale=True, num_jobs=-1, cluster_job_id=None):
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

        design = self.rescale_design_to_cpm(design)
        x = design[0]
        y = design[1]
        r = np.sqrt(x*x + y*y) + eps
        z = -1*np.sin(r)/r
        print(f"({x}, {y}) => {z}")
        return [z], responses_dict