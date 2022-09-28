from abc import ABC, abstractmethod
#from typing import *
from cmath import inf
import os
import pandas as pd
import numpy as np
import cpm
from scipy.optimize import curve_fit
from scipy import power, arange, random, nan, interpolate
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass

class OptimizerCPM:
    def __init__(self, run_folder, case_fname, fit_param_fname, recipe_fname, info_fname):
        self.name = "Optimizer"
        self._run_folder = run_folder
        self._case_fname = case_fname
        self._fit_param_fname = fit_param_fname
        self._recipe_fname = recipe_fname
        self._info_fname = info_fname
        self.exp_data = None

    def get_cyclic_fits(self, cases, cyclic_case_ids):
        """ Get cycles data """
        dat = cases.loc[cases['CaseIdentifier'].isin(cyclic_case_ids), ['CaseNo', 'FilePath']]

        m = {}
        for index, row in dat.iterrows():
            f = row['FilePath']
            num = row['CaseNo']

            full_f = os.path.join(self._run_folder, f)
            m[num] = cpm.CPM.read_from_vps_cout(full_f)

        return m


    def parse_info(self, info, cases):
        """ Port of readInfoFile(...) from Matlab 
            Port of:
            for i = 1:length(cyclicCaseIDs)
                info.cyclicFits{caseNo(i),1} = importVPSCout(['FittingDataFiles/',cyclicCaseIDs{i},'.in'],0);
            end
        """
        cyclic_case_ids = cases.loc[cases['IsCyclic']
                                    == True, 'CaseIdentifier'].to_list()
        cyclic_fits = None
        
        if len(cyclic_case_ids) > 0:
            print(f"There are cycles: {len(cyclic_case_ids)} out of {len(cases)}.")
            # Get cyclic fits
            cyclic_fits = self.get_cyclic_fits(cases, cyclic_case_ids)
            
            assert type(cyclic_fits) is dict, f"All the cycles to fit should be in a dictionary"
            assert type(list(cyclic_fits.keys())[0]) is int, f"Should be a number"
            assert cyclic_fits[list(cyclic_fits.keys())[0]].shape[0] > 0, f"Not a valid cyclic curve in dim 1"
            assert cyclic_fits[list(cyclic_fits.keys())[0]].shape[1] == 2, f"Not a valid cyclic curve in dim 2"

        return {'cyclic_fits': cyclic_fits}

    def read_recipe_file(self):
        text = None
        with open(os.path.join(
                self._run_folder, *self._recipe_fname), 'r') as f:
            text = f.read()
        return text

    def read_fit_params(self):
        return pd.read_csv(os.path.join(
            self._run_folder, *self._fit_param_fname))

    def fit_curve(self, X, Y):
        """ Fit a curve
        """
        s = UnivariateSpline(X, Y, s=30)
        return s