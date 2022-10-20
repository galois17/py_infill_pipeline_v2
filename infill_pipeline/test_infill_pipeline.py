import unittest
from infill_pipeline import *
import config
import yaml

class TestInfillPipeline(unittest.TestCase):

    def test_instance(self):
        with open("unittests_data/test_config.yaml", 'r') as stream:
            config.data_loaded = yaml.safe_load(stream)

        infill = InfillPipeline('123', 
                                config.data_loaded['cpm']['lower'],
                                config.data_loaded['cpm']['upper'],
                                config.data_loaded['cpm']['pickle_folder'],
                                config.data_loaded['system']['run_folder'],
                                config.data_loaded['system']['sqlite_db'],
                                config.data_loaded['infill']
                                )
        self.assertTrue(infill != None)

    def test_atomic(self):
        infill1 = InfillPipeline('123', 
                                 config.data_loaded['cpm']['lower'],
                                 config.data_loaded['cpm']['upper'],
                                 config.data_loaded['cpm']['pickle_folder'],
                                 config.data_loaded['system']['run_folder'],
                                 config.data_loaded['system']['sqlite_db'],
                                 config.data_loaded['infill']
                                 )
        infill2 = InfillPipeline('1234',                                 
                                 config.data_loaded['cpm']['lower'],
                                 config.data_loaded['cpm']['upper'],
                                 config.data_loaded['cpm']['pickle_folder'],
                                 config.data_loaded['system']['run_folder'],
                                 config.data_loaded['system']['sqlite_db'],
                                 config.data_loaded['infill']
                                 )
        infill3 = InfillPipeline('1234',
                                 config.data_loaded['cpm']['lower'],
                                 config.data_loaded['cpm']['upper'],
                                 config.data_loaded['cpm']['pickle_folder'],
                                 config.data_loaded['system']['run_folder'],
                                 config.data_loaded['system']['sqlite_db'],
                                 config.data_loaded['infill']
                                 )

        print("Instance 1: %d" % (infill1.infill_count))
        infill1.increment_infill_count()
        print("Instance 2: %d" % (infill2.infill_count))
        infill2.increment_infill_count()
        print("Instance 3: %d" % (infill3.infill_count))

        self.assertTrue(infill1.infill_count == infill2.infill_count)
        self.assertTrue(infill2.infill_count == infill3.infill_count)
