
from abc import ABC, abstractmethod
import copy
import utility

class BlackBox:

    def __init__(self):
        self.name = "BlackBox"

    @abstractmethod
    def setup(self, should_rerun_epsc_copy=True, should_refit_exp_data=True):
        """ Set it up... """
        raise NotImplementedError

    def join_fixed_values_to_design(self, design):
        """ Join fixed values to the design if there any.
        Args:
            design: a list
        """
        if len(self._lower) == len(design):
            # No need to add fixed params
            return design

        c_design = copy.deepcopy(design)
        fixed_values = utility.fixed_bound(self._lower, self._upper)
        new_design = [0]*(len(design) + len(fixed_values))
        j = 0
        while True:
            if j in fixed_values:
                new_design[j] = self._lower[j]
            else:
                if len(c_design) == 0:
                    break
                new_design[j] = c_design.pop(0)
            j += 1
        
        return new_design

    def rescale_design_to_cpm(self, full_design, fixed_values=None):
        c_design = copy.deepcopy(full_design)
        if not fixed_values:
            fixed_values = utility.fixed_bound(self._lower, self._upper)

        for j in range(0, len(c_design)):
            if j in fixed_values:
                continue
            else:
                l = self._lower[j]
                u = self._upper[j]
                if c_design[j] > 1 or c_design[j] < 0:
                    raise Exception('The design should have parameter values between 0 and 1.')
                c_design[j] = utility.rescale_val(c_design[j], 0, 1, l, u)
        return c_design
        
    @abstractmethod
    def obj_fun(self, design, is_initial_design_point, with_metadata=False, infill_id=None, rescale=True, num_jobs=-1, cluster_job_id=None):
        """ Objective function"""
        raise NotImplementedError

    def __str__(self):
        return f'BlackBox is named {self.name}'

    def __repr__(self):
        return f'BlackBox(name={self.name})'
