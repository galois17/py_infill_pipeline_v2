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
    def __init__(self, run_id, run_folder, design_csv_file, design_number_of_obs, case_ids, subprocess_timeout, settings, optim):
        super().__init__(run_id, run_folder, design_csv_file, design_number_of_obs, case_ids, subprocess_timeout, settings, optim)
        self.name = "CPMSS316L"
        self.__num_of_vf_response_to_include = settings['num_of_vf_response_to_include']

    def __str__(self):
        return f"CPMSS316L run_id is {self._run_id}, run_folder is {self._run_folder}, design_csv_file is {self._design_csv_file}"
