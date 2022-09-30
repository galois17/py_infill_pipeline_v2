
from abc import ABC, abstractmethod

class BlackBox:

    def __init__(self):
        self.name = "BlackBox"

    @abstractmethod
    def setup(self, should_rerun_epsc_copy=True, should_refit_exp_data=True):
        """ Set it up... """
        raise NotImplementedError

    @abstractmethod
    def obj_fun(self, design, is_initial_design_point, with_metadata=False, infill_id=None, rescale=True, num_jobs=-1, cluster_job_id=None):
        """ Objective function"""
        raise NotImplementedError

    def __str__(self):
        return f'BlackBox is named {self.name}'

    def __repr__(self):
        return f'BlackBox(name={self.name})'
