import unittest
from infill_pipeline import *
import config
import yaml
from optimizer_DP780 import OptimizerDP780
import pandas as pd

class TestOptimizerDP780(unittest.TestCase):
    def setup(self):
        with open("unittests_data/test_config.yaml", 'r') as stream:
            config.data_loaded = yaml.safe_load(stream)

        self.optim = OptimizerDP780(config.data_loaded['system']['run_folder'], config.data_loaded['cpm']
                                ['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname'])

    def test_instance(self):
        self.assertTrue(True)

    def test_read_recipe_file(self):
        self.setup()
        #read_fit_params
        df = self.optim.read_fit_params()
        #recipe_text = self.optim.read_recipe_file(config.data_loaded['cpm']['recipe_fname'], df.size)
        recipe_text = self.optim.read_recipe_file()
        print("\n")
        print("Get line\n")
        print("\t" + recipe_text.splitlines()[2])
        print("\n")

        self.optim.setup()
        self.assertTrue(self.optim.fit_recipe == "2 3 4 5 6 7 8 9 10 11 14 15 16 17 18")
