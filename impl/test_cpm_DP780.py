import unittest
from unittest import mock
from impl.cpm_DP780 import CPMDP780
import config
import yaml
import tempfile
import pandas as pd
from impl.optimizer_DP780 import OptimizerDP780
from unittest.mock import Mock 
from cpm import CPM

class TestCPMDP780(unittest.TestCase):
    def setup(self, optim=None):
        with open("unittests_data/test_config.yaml", 'r') as stream:
            config.data_loaded = yaml.safe_load(stream)
        self.inst = CPMDP780(123, config.data_loaded['system']['run_folder'],
                              config.data_loaded['infill']['r_design_init_out_file'],
                              config.data_loaded['infill']['design_num_obs'],
                              config.data_loaded['infill']['case_ids'],
                              config.data_loaded['system']['subprocess_timeout'],
                              config.data_loaded['cpm'],
                              optim)

    
    def test_instance(self):
        self.setup()
        self.assertTrue(self.inst != None)

    def test_read_vps_cout(self):
        self.setup()
        path_to_file = './unittests_data/output 1.out'
        res = CPM.read_from_vps_cout(path_to_file)
        print(res)
        print(res.shape)
        self.assertTrue(res.shape[0] > 0)

    def test_obj_fun(self):
        optim = OptimizerDP780(config.data_loaded['system']['run_folder'], 
            config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname']
            )
        self.setup(optim)

        # Setup the mocks
        dat = []
        # Mock out osome functions
        optim.error_eval_wrap = Mock(return_value=[1, 2, 1])
        optim.case_fname_as_df = Mock(return_value="")
        self.inst.write_out_dps_xfile_and_execute = Mock(return_value=("1", dat))
        
        response, responses_dict = self.inst.obj_fun([], False, num_jobs=1)
        self.assertTrue(response == [1, 2, 1])
        optim.error_eval_wrap.assert_called_once()
        self.assertTrue(self.inst.write_out_dps_xfile_and_execute.call_count == 9)

    @mock.patch("cpm.os")
    @mock.patch("cpm.shutil")
    def test_setup(self, mock_os, mock_shutil):
        optim = OptimizerDP780(config.data_loaded['system']['run_folder'], 
            config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname']
            )
        self.setup(optim)
        mock_shutil.copytree = Mock(return_value="")

        self.assertIsNone(self.inst.setup(should_rerun_epsc_copy=False))

    def test_setup_copy_epsc(self):
        optim = OptimizerDP780(config.data_loaded['system']['run_folder'], 
            config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname']
            )
        self.setup(optim)
        
        with mock.patch('cpm.shutil.copytree', return_value='') as ff:
            # Partial mock
            self.inst.setup(should_rerun_epsc_copy=True)
            ff.assert_called()
        