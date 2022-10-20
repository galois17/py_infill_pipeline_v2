from cmath import inf
import os
import pandas as pd
import numpy as np
import impl.cpm_SS316L
import scipy
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy import power, arange, random, nan, interpolate
from dataclasses import dataclass
import yaml
import pickle
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import config
import utility
from optimizer_CPM import OptimizerCPM

class OptimizerSS316L(OptimizerCPM):
    def __init__(self, run_folder, case_fname, fit_param_fname, recipe_fname, info_fname):
        super().__init__(run_folder, case_fname, fit_param_fname, recipe_fname, info_fname)
        self.name = "OptimizerSS316L"

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
        
        if self.exp_data is None:
            self.setup_exp_data()

    def setup_exp_data(self, should_reload=False):
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
                    
    def get_cyclic_fits(self, cases, cyclic_fits):
        raise NotImplementedError
 
    def prep_exp_data(self, cases, cyclic_fits):
        """ Get expData from datafiles specified in FilePath--
        e.g. 'FittingDataFiles/SR1_1e_3.txt'.

        Basically, this is prepData(...) from the original codebase.

        CaseNo                FilePath                 CaseIdentifier         SimOut         Column_x    Column_y    Start     End     Dataset    DatasetIdentifier    PlotFigure    IsCyclic    IsPF    PFRange      PFFileName       FittingWeight    description 
        ______    _________________________________    ______________    ________________    ________    ________    _____    _____    _______    _________________    __________    ________    ____    _______    _______________    _____________    ____________

        1       {'FittingDataFiles/SR1_1e_3.txt'}     {'SR1_1e_3'}     {'output 1.out'}       1           7            0    0.048       1            {'PT'}              1            0         0         0       {'TEX0001.OUT'}          1          {'SR1_1e_3'}
        2       {'FittingDataFiles/SR2_1e_4.txt'}     {'SR2_1e_4'}     {'output 1.out'}       1           7        0.048    0.093       1            {'PT'}              2            0         0         0       {'TEX0001.OUT'}          1          {'SR2_1e_4'}
        3       {'FittingDataFiles/SR3_1e_2.txt'}     {'SR3_1e_2'}     {'output 1.out'}       1           7        0.093    0.137       1            {'PT'}              3            0         0         0       {'TEX0001.OUT'}          1          {'SR3_1e_2'}
        4       {'FittingDataFiles/SR4_5e_3.txt'}     {'SR4_5e_3'}     {'output 1.out'}       1           7        0.137    0.178       1            {'PT'}              4            0         0         0       {'TEX0001.OUT'}          1          {'SR4_5e_3'}
        5       {'FittingDataFiles/SR5_1e_1.txt'}     {'SR5_1e_1'}     {'output 1.out'}       1           7        0.178      0.4       1            {'PT'}              5            0         0         0       {'TEX0001.OUT'}          1          {'SR5_1e_1'}
        6       {'FittingDataFiles/SS_n15.txt'  }     {'Tempn15' }     {'output 1.out'}       1           7            0     0.55       1            {'PT'}              6            0         0         0       {'TEX0001.OUT'}          1          {'Tempn15' }
        7       {'FittingDataFiles/SS_0.txt'    }     {'Temp0'   }     {'output 1.out'}       1           7            0     0.55       1            {'PT'}              7            0         0         0       {'TEX0001.OUT'}          1          {'Temp0'   }
        8       {'FittingDataFiles/SS_10.txt'   }     {'Temp10'  }     {'output 1.out'}       1           7            0     0.55       1            {'PT'}              8            0         0         0       {'TEX0001.OUT'}          1          {'Temp10'  }
        9       {'FittingDataFiles/SS_20.txt'   }     {'Temp20'  }     {'output 1.out'}   
        """

        data_files = cases.loc[:, 'FilePath'].to_list()
        X = [None]*len(data_files)
        smoothed_Y = [None]*len(data_files)
        exp_y_fcn =  [None]*len(data_files)
        raw_Y =  [None]*len(data_files)

        for j in range(0, len(data_files)):
            out = impl.cpm_SS316L.CPMSS316L.read_from_vps_cout(os.path.join(self._run_folder, data_files[j]))
            X[j] = out[:, 0]
            # experimental curve y data
            cur_exp_y = out[:, 1]
            # Make a copy of the raw y data
            raw_Y[j] = np.copy(cur_exp_y)

            if cases.loc[j, 'IsCyclic'] == 1:
                raise NotImplementedError
            else:
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
                    print(f"Oops at {j}\n")
                    raise e

        return {'X': X, 'smoothed_Y': smoothed_Y, 'raw_Y': raw_Y, 'exp_y_fcn': exp_y_fcn}

    def error_eval_wrap(self, cases, sim_data_dict):
        """ Error wrap 
        Returns:
            An np array
        """
        errors = [0]*cases.shape[0]

        fitting_weight = cases.loc[:, 'FittingWeight']
        fitting_weight = fitting_weight.to_list()

        if self.exp_data is None:
            self.setup_exp_data()
        
        vf_exp1 = pd.read_csv(os.path.join(self._run_folder, 'VF1.csv'))
        vf_exp2 = pd.read_csv(os.path.join(self._run_folder, 'VF2.csv'))
        vf_exp3 = pd.read_csv(os.path.join(self._run_folder, 'VF3.csv'))
        vf_exp4 = pd.read_csv(os.path.join(self._run_folder, 'VF4.csv'))

        vf_exp = [vf_exp1, vf_exp2, vf_exp3, vf_exp4]
        
        errors = [0]*(cases.shape[0]+4)
        for j in range(0, cases.shape[0]):
            # Determine the case, regular cases are stress-strain curves, extended
            # cases are pole figures etc.
            exp_x = self.exp_data['X'][j]
            #exp_y_fcn = self.exp_data['exp_y_fcn'][j]
            exp_y_fcn = self.exp_data['raw_Y'][j] 
            # Get the actual case number (1 index)
            case_no = j+1
            
            # Get the sim data for case j
            sim_data = sim_data_dict[str(case_no)]

            if sim_data is None:
                errors[j] = 1000
                continue

            sim_modx = sim_data[:,0]
            sim_mody = sim_data[:,1]
            sim_VF = sim_data[:,2]

            assert (sim_modx.dtype.char in np.typecodes['AllFloat']), f"sim_modx needs to be a float type, instead it's a {sim_modx.dtype}"
            assert (sim_mody.dtype.char in np.typecodes['AllFloat']), f"sim_mody needs to be a float type, instead it's a {sim_mody.dtype}"
            assert (exp_x.dtype.char in np.typecodes['AllFloat']), f"exp_x needs to be a float type, instead it's a {exp_x.dtype}"

            err, _ = self.calc_error_piece_wise(exp_x, exp_y_fcn, sim_modx, sim_mody, None)
            
            errors[j] = err
            # Error for VF of temperature cases, 6, 7, 8, 9
            if j >= 5:
                # Find the VF at specific strains
                err_ind = 0
                target_strain = vf_exp[j-5].iloc[err_ind, 0]
                target_vf = vf_exp[j-5].iloc[:,1]
                num_data = vf_exp[j-5].shape[0]
                enum_sim_vf = np.zeros((num_data,1))
                x = np.array([])
                
                target_vf_comp = np.zeros((num_data,1))
                for kk in range(0, len(sim_modx)):
                    if sim_modx[kk] >= target_strain:
                        x = np.append(x, sim_modx[kk])
                        enum_sim_vf[err_ind] = sim_VF[kk]
                        target_vf_comp[err_ind] = float(target_vf.values[err_ind])

                        err_ind += 1
                        if err_ind >= num_data:
                            break
                        target_strain = vf_exp[j-5].iloc[err_ind, 0]

                errors[j+4] = np.sqrt(np.sum(np.square(enum_sim_vf - target_vf_comp))/len(enum_sim_vf))
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
            try:
                x, y, z = self.interpolate(0, len(sim_modx)-1, 0, len(exp_x)-1, sim_modx, sim_mody, exp_x, exp_fcn)
            except Exception as e:
                return (2000+np.random.normal(0, 100)), 0

            pieces_x = [x]
            pieces_sim_y = [y]
            pieces_exp_y = [z]

        return compute_rmse_for_pieces(pieces_sim_y, pieces_exp_y), len(pieces_x)
