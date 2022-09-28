import unittest

from infill_pipeline import *
from blackbox import *

class TestCPM(unittest.TestCase):

    def test_instance(self):
        bb = BlackBox()
        self.assertTrue(bb != None)
