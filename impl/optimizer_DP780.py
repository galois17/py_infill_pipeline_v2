import os
import sys
from pathlib import Path
import scipy
import yaml
import pickle
from scipy.optimize import curve_fit
from scipy import power, arange, random, nan, interpolate
from dataclasses import dataclass
import pandas as pd
import numpy as np
from cmath import inf
import similaritymeasures as sm
import impl.cpm_DP780

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import config
import utility
from optimizer_CPM import OptimizerCPM

class OptimizerDP780(OptimizerCPM):
    """ Optimizer for DP780
    """
    def __init__(self, run_folder, case_fname, fit_param_fname, recipe_fname, info_fname):
        super().__init__(run_folder, case_fname, fit_param_fname, recipe_fname, info_fname)
        self.name = "OptimizerDP780"

    def setup(self):
        # Reading in the csv file containing information on the experimental data
        # and where to find the simulation results
        self.cases = pd.read_csv(os.path.join(
            self._run_folder, *self._case_fname))

        # Reading in the csv file containing information on the parameters to be
        # written and fitted for each generation. Also contains information on
        # scaling, lb, ub for GA input, fitFlag, and more, see example file
        self.fit_params = self.read_fit_params()

        # Read the recipe file. The file is structured as:
        # Specify the recipe to fit
        # 3
        # Recipes:
        # 2 3 4 5 6 7 8 9 10 11 14 15 16 17
        # 14 15 16 17
        # 2 3 4 5 6 7 8 9 10 11 14 15 16 17 18
        #
        # Each row below "Recipes:", describes a recipe.
        # A missing number is a parameter that is fixed constant.
        # In the above, for recipe three, 1, 12, and 13 are fixed since
        # they are missing.
        fit_recipe_text = self.read_recipe_file()
        print(fit_recipe_text)
        recipe_num = int(fit_recipe_text.splitlines()[1])
        self.fit_recipe = fit_recipe_text.splitlines()[recipe_num + 2]

        # Reading in the text file containing various other information such as
        # type of operating system and the name of the executable of the model to
        # be run. ishift flag for considering a shifted curve in the errors.
        # Also contains all the other GA inputs.

        case_fname = config.data_loaded['cpm']['case_fname']
        run_folder = config.data_loaded['system']['run_folder']
        path_to_case_fname = os.path.join(run_folder, *case_fname)
        self.case_fname_as_df = pd.read_csv(path_to_case_fname)

        config_info = None
        with open(os.path.join(self._run_folder, *self._info_fname), 'r') as stream:
            config_info = yaml.safe_load(stream)

        parse_info_res = self.parse_info(config_info, self.case_fname_as_df)

        # Get the cyclic fits
        self.cyclic_fits = parse_info_res['cyclic_fits']

        if self.exp_data is None:
            self.setup_exp_data()

    def setup_exp_data(self, should_reload=False):
        """ Setup experimental data. 

        Args: 
            should_reload: True to reload data from raw files; False to load from Pickle
        """
        case_fname = config.data_loaded['cpm']['case_fname']
        run_folder = config.data_loaded['system']['run_folder']
        path_to_case_fname = os.path.join(run_folder, *case_fname)
        case_fname_as_df = pd.read_csv(path_to_case_fname)

        config_info = None
        with open(os.path.join(self._run_folder, *self._info_fname), 'r') as stream:
            config_info = yaml.safe_load(stream)
        parse_info_res = self.parse_info(config_info, case_fname_as_df)
        cyclic_fits = parse_info_res['cyclic_fits']

        if should_reload:
            self.exp_data = self.prep_exp_data(case_fname_as_df, cyclic_fits)
            # Save the exp data
            with open(os.path.join(self._run_folder, 'exp_data.pickle'), 'wb') as handle:
                pickle.dump(self.exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if os.path.exists(os.path.join(self._run_folder, 'exp_data.pickle')):
                with open(os.path.join(self._run_folder, 'exp_data.pickle'), 'rb') as handle:
                    self.exp_data = pickle.load(handle)
            else:
                # The pickle does not exist, need to create it
                self.exp_data = self.prep_exp_data(case_fname_as_df, cyclic_fits)
                # Save the exp data
                with open(os.path.join(self._run_folder, 'exp_data.pickle'), 'wb') as handle:
                    pickle.dump(self.exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def prep_exp_data(self, cases, cyclic_fits):
        """ Prep the experimental data.
        Get expData from datafiles specified in FilePath--
        e.g. 'FittingDataFiles/SR1_1e_3.txt'.

        Basically, this is prepData(...) from the original codebase. 
        """

        data_files = cases.loc[:, 'FilePath'].to_list()
        X = [None]*len(data_files)
        smoothed_Y = [None]*len(data_files)
        exp_y_fcn =  [None]*len(data_files)
        raw_Y =  [None]*len(data_files)
        segmentations = [None]*len(data_files)

        for j in range(0, len(data_files)):
            out = impl.cpm_DP780.CPMDP780.read_from_vps_cout(os.path.join(self._run_folder, data_files[j]))
            
            # Experimental curve x data
            X[j] = np.copy(out[:, 0])
            # experimental curve y data
            cur_exp_y = out[:, 1]
            # Make a copy of the raw y data
            raw_Y[j] = np.copy(cur_exp_y)

            if cases.loc[j, 'IsCyclic'] == 1:
                smoothed_Y[j] = utility.smooth(np.copy(cur_exp_y), 3)

                # Find segmentation file
                seg_file = data_files[j]
                seg_file = Path(seg_file).stem
                seg_file = f"{seg_file}.in"
                full_path_seg_file = os.path.join(self._run_folder, "FittingDataFiles", seg_file)

                out_segmentation = impl.cpm_DP780.CPMDP780.read_from_vps_cout(full_path_seg_file)
                segmentations[j] = out_segmentation
            else:
                segmentations[j] = None
                smoothed_Y[j] = utility.smooth(cur_exp_y, 3)
                
                try:
                    sort_idx = np.argsort(X[j])
                    sorted_X = (X[j])[sort_idx]
                    sorted_smoothed_Y = (smoothed_Y[j])[sort_idx]
                    u, indices = np.unique(sorted_X, return_index=True)
                    X_unique = sorted_X[indices]
                    smoothed_Y_unique = sorted_smoothed_Y[indices]
                    exp_y_fcn[j] = self.fit_curve(X_unique, smoothed_Y_unique)
                    X[j] = X_unique
                    utility.log(f"\nFitting curve for case {j} (0 indexed)\n")
                except ValueError as e:
                    print(f"Unable to use the curve at {j}\n")
                    print(e)
                    raise e

        return {'X': X, 'smoothed_Y': smoothed_Y, 'exp_y_fcn': exp_y_fcn, 'raw_Y': raw_Y, 'segmentations': segmentations}

    def error_eval_wrap(self, cases, sim_data_dict):
        """ Error wrap 
        Returns:
            An np array
        """
        errors = [0]*cases.shape[0]

        fitting_weight = cases.loc[:, 'FittingWeight']
        fitting_weight = fitting_weight.to_list()

        fit_range_rows = cases.loc[:, ['Start', 'End']]
        # list of list or list of ranges
        fit_range = fit_range_rows.values.tolist()

        output_filename_rows = cases.loc[:, 'SimOut']
        output_filenames = output_filename_rows.to_list()

        col_x_rows = cases.loc[:, 'Column_x']
        col_x = col_x_rows.to_list()

        col_y_row = cases.loc[:, 'Column_y']
        col_y = col_y_row.to_list()

        data_files_row = cases.loc[:, 'FilePath']
        data_files = data_files_row.to_list()

        if self.exp_data is None:
            self.setup_exp_data()

        errors = [0]*(cases.shape[0])
        for j in range(0, cases.shape[0]):
            exp_x = self.exp_data['X'][j]
            exp_y = self.exp_data['smoothed_Y'][j]
            #cur_fitting_weight = fitting_weight[j]
            cur_fit_range = fit_range[j]
            exp_y_fcn = self.exp_data['raw_Y'][j]            
            segmentation = self.exp_data['segmentations'][j]

            # Get the actual case number (1 index)
            case_no = j+1
            # Get the sim data for case j
            sim_data = sim_data_dict[str(case_no)]
            if sim_data is None:
                errors[j] = 1000
                continue 
            sim_modx = sim_data[:,0]
            sim_mody = sim_data[:,1]

            assert (sim_modx.dtype.char in np.typecodes['AllFloat']), f"sim_modx needs to be a float type, instead it's a {sim_modx.dtype}"
            assert (sim_mody.dtype.char in np.typecodes['AllFloat']), f"sim_mody needs to be a float type, instead it's a {sim_mody.dtype}"
            assert (exp_x.dtype.char in np.typecodes['AllFloat']), f"exp_x needs to be a float type, instead it's a {exp_x.dtype}"
            assert (exp_y_fcn.dtype.char in np.typecodes['AllFloat']), f"exp_y_fcn needs to be a float type, instead it's a {exp_y_fcn.dtype}"
            print(f"Finding error for case {case_no}")
            err, _ = self.calc_error_piece_wise(exp_x, exp_y_fcn, sim_modx, sim_mody, segmentation)
            
            if np.isnan(err):
                raise Exception("Should not be nan...")
            errors[j] = err  
        return errors

    def calc_error_piece_wise(self, exp_x, exp_fcn, sim_modx, sim_mody, segmentation):
        """ Calculate the error piece-wise
        Args:

        Returns:
            RMSE 
        """
    
        def compute_rmse_for_pieces(y1, y2):
            tot = 0
            tot_length = 0
            for j in range(len(y1)):
                tot +=  np.sum((y2[j]-y1[j])**2)
                tot_length += len(y1[j])
            if tot_length == 0:
                return 1000 
            return np.sqrt(tot/tot_length)

        pieces_x, pieces_sim_y, pieces_exp_y = self.get_piecewise(sim_modx, sim_mody, exp_x, exp_fcn, segmentation)
        if segmentation is None:
            # No cycles
            x, y, z = self.interpolate(0, len(sim_modx)-1, 0, len(exp_x)-1, sim_modx, sim_mody, exp_x, exp_fcn)
            pieces_x = [x]
            pieces_sim_y = [y]
            pieces_exp_y = [z]
            
        return compute_rmse_for_pieces(pieces_sim_y, pieces_exp_y), len(pieces_x)

