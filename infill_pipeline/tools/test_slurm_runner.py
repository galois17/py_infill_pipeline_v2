import unittest
import utility
import re
import datetime
import os
from numpy import *
from unittest.mock import Mock
from unittest import mock
from slurm_runner import SlurmRunner
import tempfile

class TestSlurmRunner(unittest.TestCase):
    #with patch.object(ProductionClass, 'method', return_value=None) as mock_method:
    def test_instantiate(self):
        slurm_runner = SlurmRunner(None, None)
        self.assertTrue(slurm_runner is not None)