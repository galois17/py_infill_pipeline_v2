from cmath import inf
import config
import csv
import os
import pandas as pd
import numpy as np
import impl.cpm_SS316L
import utility
from scipy.optimize import curve_fit
from scipy import power, arange, random, nan, interpolate
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
import yaml
import pickle
import math
import traceback
import sys

class OptimizerSimpleCase:
    def __init__(self, run_folder):
        pass