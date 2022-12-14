import os
import sys
from abc import ABC, abstractmethod
from joblib import delayed

from infill_pipeline.cpm import CPM

class CPMDP780(CPM):
    """ DP780 Experiment """

    def __init__(self, run_id, run_folder, design_csv_file, design_number_of_obs, selected_cases, case_ids, subprocess_timeout, settings, optim):
        super().__init__(run_id, run_folder, design_csv_file, design_number_of_obs, selected_cases, case_ids, subprocess_timeout, settings, optim)
        self.name = "CPMDP780"

    def __str__(self):
        return f"CPMDP780 run_id is {self._run_id}, run_folder is {self._run_folder}, design_csv_file is {self._design_csv_file}"
