import unittest
from unittest import mock
from impl.cpm_SS316L import *
import config
import yaml
import tempfile
import pandas as pd
from impl.optimizer_SS316L import OptimizerSS316L
from unittest.mock import Mock 

class TestCPMSS316L(unittest.TestCase):
    def setup(self, optim=None):
        with open("unittests_data/test_config.yaml", 'r') as stream:
            config.data_loaded = yaml.safe_load(stream)
        self.inst = CPMSS316L(123, config.data_loaded['system']['run_folder'],
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

    def test_write_dps_xfile(self):
        self.setup()

        with tempfile.NamedTemporaryFile(suffix='.html', prefix=os.path.basename(__file__)) as tf:
            tf.write(b"Some data {{ par[0] | filter_string_format: '%.2f' }} and {{ par[1] }}")
            tf.flush()

            filename = tf.name
            print("In here...")
            print(filename)
            res = self.inst.write_dps_xfile(filename, [2.12345, [1, 2, 3]])
            print(res)
            self.assertTrue(res == "Some data 2.12 and [1, 2, 3]")
    
    def test_rescale_with_fixed_param(self):
        self.setup()

        d = [1e8,  0.1, 1e2, 100,  0.01,  200, 1e8, 0.05, 2e2, 100, 5, -100, 0.1, 0.1, 0.01]
        full_design = self.inst.join_fixed_values_to_design(d)

        self.assertTrue(full_design[0] == 0.9)
        self.assertTrue(full_design[11] == 0.0)
        self.assertTrue(full_design[12] == 1000000.0)

    def test_obj_fun(self):
        optim = OptimizerSS316L(config.data_loaded['system']['run_folder'], 
            config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname']
            )
        self.setup(optim)

        # Setup the mocks
        dat = []
        optim.error_eval_wrap = Mock(return_value=[1, 2, 1])
        optim.case_fname_as_df = Mock(return_value="")
        self.inst.write_out_dps_xfile_and_execute = Mock(return_value=("1", dat))
        
        response, responses_dict = self.inst.obj_fun([], False, num_jobs=1)
        self.assertTrue(response == [1, 2, 1])
        optim.error_eval_wrap.assert_called_once()
        self.assertTrue(self.inst.write_out_dps_xfile_and_execute.call_count == 9)

    def test_setup_copy_epsc(self):
        optim = OptimizerSS316L(config.data_loaded['system']['run_folder'], 
            config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname']
            )
        self.setup(optim)
        
        with mock.patch('cpm.shutil.copytree', return_value='') as ff:
            # Partial mock
            self.inst.setup(should_rerun_epsc_copy=True)
            ff.assert_called()