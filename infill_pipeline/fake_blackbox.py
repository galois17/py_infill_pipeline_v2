from abc import ABC, abstractmethod

from infill_pipeline.blackbox import BlackBox

class FakeBlackbox(BlackBox):
    def __init__(self):
        super().__init__()
        self.name = "FakeBlackbox"

    def setup(self):
        pass

    def obj_fun(self, design, with_metadata=False, infill_id=None, cluster_job_id=None):
        pass