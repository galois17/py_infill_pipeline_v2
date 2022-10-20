import unittest
import yaml
import pandas as pd
import os
import numpy as np

from infill_pipeline.infill_pipeline import InfillPipeline
import infill_pipeline.config as config
from infill_pipeline.impl.optimizer_SS316L import OptimizerSS316L

class TestOptimizerSS316L(unittest.TestCase):
    def setup(self):
        with open("unittests_data/test_config.yaml", 'r') as stream:
            config.data_loaded = yaml.safe_load(stream)

        self.optim = OptimizerSS316L(config.data_loaded['system']['run_folder'], config.data_loaded['cpm']
                                ['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname'])

    def test_instance(self):
        self.assertTrue(True)

    def test_setup(self):
        self.setup()

        self.optim.setup()
        print(f"Size of cases is {self.optim.cases.shape[0]}")
        self.assertTrue(self.optim.cases.shape[0] == 9)

        for j in range(1, 10):
            row = self.optim.cases.loc[self.optim.cases['CaseNo'] == j]
            filename = row['FilePath'].item()

            # Check if file exists
            fullpath_to_file = os.path.join(
                config.data_loaded['system']['run_folder'], filename)
            # Check to see that the file exists
            self.assertTrue(os.path.isfile(fullpath_to_file))

        self.assertTrue(self.optim.fit_params.shape[0] > 0)
        self.assertTrue(self.optim.fit_params.shape[1] == 8)

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

    def test_parse_info(self):
        self.setup()
        run_folder = config.data_loaded['system']['run_folder']
        config_info = None
        with open(os.path.join(run_folder, "Inp_Info_PT.yaml"), 'r') as stream:
            config_info = yaml.safe_load(stream)

        self.assertTrue(config_info is not None)
        optim = OptimizerSS316L(config.data_loaded['system']['run_folder'], config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname'])
        case_fname = config.data_loaded['cpm']['case_fname']
        run_folder = config.data_loaded['system']['run_folder']
        path_to_case_fname = os.path.join(run_folder, *case_fname)
        case_fname_as_df = pd.read_csv(path_to_case_fname)

        res = None
        try:
            res = optim.parse_info(config_info, case_fname_as_df)
        except Exception:
            self.fail("Should not have raised an exception when there's no cyclic feature");

        self.assertIsNone(res['cyclic_fits'])

    def test_read_vps_cout(self):
        self.setup()
        optim = OptimizerSS316L(config.data_loaded['system']['run_folder'], config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname'])

        case_fname = config.data_loaded['cpm']['case_fname']
        run_folder = config.data_loaded['system']['run_folder']
        path_to_case_fname = os.path.join(run_folder, *case_fname)
        case_fname_as_df = pd.read_csv(path_to_case_fname)

        self.assertTrue(case_fname_as_df.size > 0)

    def test_obj_fun(self):
        self.setup()
    
        self.assertTrue(True)

    def test_fit_curve(self):
        optim = OptimizerSS316L(config.data_loaded['system']['run_folder'], config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname'])
        x = np.arange(-10, 10, 0.1)
        y = 3 * np.exp(-0.05*x) + 12 + 1.4 * np.sin(1.2*x) + 2.1 * np.sin(-2.2*x + 3)
        y_noise = y + np.random.normal(0, 0.5, size = len(y))
        try:
            y_fitted = optim.fit_curve(x, y_noise)
        except Exception:
            self.fail("Should not have raised an exception")

    @unittest.skip("takes too long...")
    def test_prep(self):
        self.setup()

        run_folder = config.data_loaded['system']['run_folder']
        config_info = None
        with open(os.path.join(run_folder, "Inp_Info_PT.yaml"), 'r') as stream:
            config_info = yaml.safe_load(stream)

        optim2 = OptimizerSS316L(config.data_loaded['system']['run_folder'], config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'])

        case_fname = config.data_loaded['cpm']['case_fname']
        run_folder = config.data_loaded['system']['run_folder']
        path_to_case_fname = os.path.join(run_folder, *case_fname)
        case_fname_as_df = pd.read_csv(path_to_case_fname)

        res = optim2.parse_info(config_info, case_fname_as_df)
        cyclic_fits = res['cyclic_fits']
        
        # TODO: fix test. Takes too long.
        try:
            optim2.prep_exp_data(case_fname_as_df, cyclic_fits)
        except Exception:
            self.fail("Should not have raised an exception")