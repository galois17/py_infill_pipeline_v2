import unittest
import utility
import re
import datetime
import os
from numpy import *
from unittest.mock import Mock
from unittest import mock
import tempfile

class TestUtility(unittest.TestCase):
    #@unittest.skip("demonstrating skipping")
    def test_run_id(self):
        id = utility.generate_run_id(datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
        print(id)
        result = re.match(r"^ID_\d+_\d+", id)
        self.assertTrue(result)

    def test_smooth(self):
        t = linspace(-4,4,100)
        x = sin(t)
        xn = x + random.randn(len(t))*0.1
        y = utility.smooth(x, 3)
        print(f"lengths {len(x)} and {len(y)}")
        print(x)
        print(y)
        self.assertTrue(len(y) == len(x))

    @mock.patch.dict(os.environ, {"CONFIG_FILE": "unittests_data/test_config.yaml"})
    def test_get_config(self):
        self.assertTrue(utility.get_config_file() == 'unittests_data/test_config.yaml')
        

