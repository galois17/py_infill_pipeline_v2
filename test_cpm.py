import unittest
from cpm import CPM
import config
import yaml
from impl.optimizer_SS316L import OptimizerSS316L

class TestCPM(unittest.TestCase):
    def setup(self, optim=None):
        with open("unittests_data/test_config.yaml", 'r') as stream:
            config.data_loaded = yaml.safe_load(stream)
        self.inst = CPM(123, config.data_loaded['system']['run_folder'],
                              config.data_loaded['infill']['r_design_init_out_file'],
                              config.data_loaded['infill']['design_num_obs'],
                              config.data_loaded['infill']['case_ids'],
                              config.data_loaded['system']['subprocess_timeout'],
                              config.data_loaded['cpm'],
                              optim)

    def test_instance(self):
        self.setup()
        self.assertTrue(self.inst != None)

    # def test_setup(self):
    #     optim = OptimizerSS316L(config.data_loaded['system']['run_folder'], 
    #         config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname']
    #         )
    #     self.setup(optim)
    #     self.assertTrue(self.inst.setup() == None)

