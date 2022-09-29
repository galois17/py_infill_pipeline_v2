from abc import ABC, abstractmethod
import os
import signal
import sys
import subprocess
import shutil
from joblib import Parallel, delayed
import numpy as np
from liquid import Liquid
import pandas as pd
import random
import platform
import pickle
import datetime
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import liquid_template_helper
import config
import utility
from cpm import CPM

class CPMSS316L(CPM):
    """ SS316L Experiment """

    def __init__(self, run_id, run_folder, design_csv_file, design_number_of_obs, selected_cases, case_ids, subprocess_timeout, settings, optim):
        super().__init__(run_id, run_folder, design_csv_file, design_number_of_obs, selected_cases, case_ids, subprocess_timeout, settings, optim)
        self.name = "CPMSS316L"

    def __str__(self):
        return f"CPMSS316L run_id is {self._run_id}, run_folder is {self._run_folder}, design_csv_file is {self._design_csv_file}"
