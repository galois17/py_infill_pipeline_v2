import unittest

from infill_pipeline.infill_pipeline import InfillPipeline
from infill_pipeline.blackbox import BlackBox

class TestCPM(unittest.TestCase):

    def test_instance(self):
        bb = BlackBox()
        self.assertTrue(bb != None)
