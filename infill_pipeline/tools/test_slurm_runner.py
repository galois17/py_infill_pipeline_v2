import unittest
import re
import datetime
import os
from numpy import *
from unittest.mock import Mock
from unittest import mock
import tempfile

from infill_pipeline.tools.slurm_runner import SlurmRunner
import infill_pipeline.utility as utility

class TestSlurmRunner(unittest.TestCase):
    #with patch.object(ProductionClass, 'method', return_value=None) as mock_method:
    def test_instantiate(self):
        slurm_runner = SlurmRunner(None, None)
        self.assertTrue(slurm_runner is not None)