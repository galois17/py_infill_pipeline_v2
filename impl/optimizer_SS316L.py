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
import traceback
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
        cyclic_fits = parse_info_res['cyclic_fits']
        
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
            print(f"The shape is {out.shape}")
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
        vf_curve = [0]*4

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
            exp_y = self.exp_data['smoothed_Y'][j]
            cur_fitting_weight = fitting_weight[j]
            cur_fit_range = fit_range[j]
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
            #assert (exp_y_fcn.dtype.char in np.typecodes['AllFloat']), f"exp_y_fcn needs to be a float type, instead it's a {exp_y_fcn.dtype}"

            is_vf = (j >= cases.shape[0])
        
            #err, _ = self.calc_error(exp_x, exp_y_fcn, cur_fit_range, sim_modx, sim_mody, sim_VF)
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
                
                for kk in range(0, len(sim_modx)):
                    if sim_modx[kk] >= target_strain:
                        x = np.append(x, sim_modx[kk])
                        enum_sim_vf[err_ind] = sim_VF[kk]
                        err_ind += 1
                        if err_ind >= num_data:
                            break
                        #print(f"Check here: {j} err_ind {err_ind} shape {vf_exp[j-5].shape}")
                        target_strain = vf_exp[j-5].iloc[err_ind, 0]
                    else:
                        x = sim_modx
                
                target_vf = target_vf.values.reshape((len(target_vf),1))
                
                errors[j+4] = np.sqrt(np.sum(np.square(enum_sim_vf - target_vf))/len(enum_sim_vf))
        return errors

    def calc_error(self, exp_x, exp_fcn, fit_range, sim_modx, sim_mody, sim_VF, is_vf_case=False):
        """ Calculate the error 

        Args:
        Returns:
            error and number of elements as a tuple
        """
        err = -1
        try:
            if fit_range[0] < fit_range[1]:
                start_x = max([fit_range[0], exp_x[0], sim_modx[0]])
                iter = 0
                
                val = np.inf
                if sim_mody[iter] != 0:
                    val = exp_fcn(sim_modx[iter])/np.float64(sim_mody[iter])

                while sim_modx[iter] < start_x or val < 0:
                    iter += 1
                    if iter >= len(sim_modx):
                        break
                start_x = iter

                end_x = min([fit_range[1], exp_x[-1], sim_modx[-1]])
                iter = len(sim_modx)-1

                val = np.inf
                if sim_mody[iter] != 0:
                    val = exp_fcn(sim_modx[iter])/sim_mody[iter]
                while sim_modx[iter] > end_x or val < 0:
                    iter -= 1
                    if iter == 0:
                        break
                
                end_x = iter
            else:
                start_x = min([fit_range[0], exp_x[0], sim_modx[0]])
                iter = 0

                val = np.inf
                if sim_mody[iter] != 0:
                    val = exp_fcn(sim_modx[iter])/sim_mody[iter]
                while sim_modx[iter] > start_x or val < 0:
                    iter += 1
                    if iter >= len(sim_modx):
                        break
                start_x = iter
                    
                end_x = max([fit_range[1], exp_x[-1], sim_modx[-1]])
                iter = len(sim_modx)-1

                val = np.inf
                if sim_mody[iter] != 0:
                    val = exp_fcn(sim_modx[iter])/sim_mody[iter]
                while sim_modx[iter] < end_x or val < 0:
                    iter -= 1
                    if iter == 0:
                        break              
                end_x = iter
            eval_x = sim_modx[start_x:(end_x+1)]
            eval_y = sim_mody[start_x:(end_x+1)]
            tot_n = len(eval_x)

            # Error evaluation based on Tofallis, 2014
            evaluted_exp_fcn = exp_fcn(eval_x)

            if eval_y.shape[0] == 0 or evaluted_exp_fcn.shape[0] == 0:
                return random.randint(0, 50), len(exp_x)

            # Compute RMSE
            err = np.sqrt(np.sum(np.square(eval_y - evaluted_exp_fcn))/tot_n)
        except Exception as e:            
            print(traceback.format_exc())
            return random.randint(0, 50), len(exp_x)
    
        return err, tot_n


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

        pieces_x, pieces_sim_y, pieces_exp_y = self.__get_piecewise(sim_modx, sim_mody, exp_x, exp_fcn, segmentation)
        if segmentation is None:
            # No cycles
            try:
                x, y, z = self.__interpolate(0, len(sim_modx)-1, 0, len(exp_x)-1, sim_modx, sim_mody, exp_x, exp_fcn)
            except Exception as e:
                return (2000+np.random.normal(0, 100)), 0

            pieces_x = [x]
            pieces_sim_y = [y]
            pieces_exp_y = [z]

        return compute_rmse_for_pieces(pieces_sim_y, pieces_exp_y), len(pieces_x)

    def __get_piecewise(self, sim_modx, sim_mody, truth_x, truth_y, segmentation):
        pieces_x = []
        pieces_actual_x = []
        pieces_sim_y =[]
        pieces_exp_y = []
        if segmentation is None:
            segmentation = [[0, len(truth_x)]]
        
        #tol = 0.008
        tol = 0.01
        track_x = 0
        for break_j in segmentation:
            print("**************************")
            print(f"Segmentation {break_j}")
            print(f"Curent tracking at {track_x}; max length {len(sim_modx)}")
            x1 = int(break_j[0])
            x2 = int(break_j[1])
            if x2 == len(truth_x):
                x2 = x2-1

            actual_exp_x1 = truth_x[x1]

            if x2 >= len(truth_x):
                actual_exp_x2 = truth_x[-1]    
            else:
                actual_exp_x2 = truth_x[x2]
            max_x = max(truth_x)
            min_x = min(truth_x)

            # Find valid segments
            all_x1 = []
            actual_all_x1 = []
            all_x2 = []
            actual_all_x2 = []

            found_interval = False
            should_quit_current_segment = False
            while track_x < len(sim_modx):
                if found_interval or should_quit_current_segment:
                    break
            
                for k in range(track_x, len(sim_modx)):
                    if k == len(sim_modx) - 1:
                        # Reached the end
                        should_quit_current_segment = True
                    
                    if abs(sim_modx[k] - actual_exp_x1)/(max_x- min_x) <= tol:                    
                        all_x1.append(k)
                        actual_all_x1.append(sim_modx[k])
                        # Find matching endpoint
                        found_matching_endpoint = False
                        for m in range(k, len(sim_modx)):
                            if abs(sim_modx[m] - actual_exp_x2)/(max_x- min_x) <= tol:
                                all_x2.append(m)
                                actual_all_x2.append(sim_modx[m])
                                found_matching_endpoint = True
                                track_x = m
                                break
                            
                        if not found_matching_endpoint:
                            # Did not find a matching endpoint
                            all_x1.pop(-1)
                            should_quit_current_segment = True
                            print("!! DID NOT FIND interval !! ")
                            print("**************************")
                            break
                        else:
                            found_interval = True
                            print(f"Found complete interval: {all_x1}, {all_x2}")
                            print("**************************")
                            break

            winner_x1 = -1
            winner_x2 = -1
            for i in range(len(all_x1)):
                pick_x1 = all_x1[i]
                pick_x2 = -1

                for j in range(len(all_x2)):
                    if all_x2[j] > pick_x1:
                        pick_x2 = all_x2[j]

                        # How much data?
                        length = pick_x2 - pick_x1
                        norm_length = length/len(sim_modx)

                        target_length = x2-x1
                        norm_target_length = target_length/len(truth_x)
                        rat = abs(norm_length-norm_target_length)*100;            

                        sim_x1_norm = pick_x1/len(sim_modx)
                        sim_x2_norm = pick_x2/len(sim_modx)

                        exp_x1_norm = x1/len(truth_x)
                        exp_x2_norm = x2/len(truth_x)

                        rat_x1 = abs(sim_x1_norm - exp_x1_norm)*100
                        rat_x2 = abs(sim_x2_norm - exp_x2_norm)*100

                        thresh_width_perc = 40
                        thresh_points_perc = 40
                        if (((truth_x[x2] > truth_x[x1]) and (sim_modx[pick_x2] > sim_modx[pick_x1])) or ((truth_x[x2] < truth_x[x1]) and (sim_modx[pick_x2] < sim_modx[pick_x1])) ) and (rat <= thresh_width_perc) and ( (rat_x1 <= thresh_points_perc) and (rat_x2 <= thresh_points_perc)  ):
                            winner_x1 = pick_x1
                            winner_x2 = pick_x2
                            print("Try to interpolate...")
                            print(f"Length of truth is {len(truth_x)}")
                            try:
                                new_xx, ynew, gnew = self.__interpolate(winner_x1, winner_x2, x1, x2, sim_modx, sim_mody, truth_x, truth_y, tol)
                            except Exception as e:
                                raise e
                            print(f"Length of x values for interpolation is {len(new_xx)}")

                            assert len(ynew) == len(gnew), "The function evaluations should have the same length."
                            
                            pieces_x.append(new_xx)
                            pieces_sim_y.append(ynew)
                            pieces_exp_y.append(gnew)
                            
        return pieces_x, pieces_sim_y, pieces_exp_y

    def __interpolate(self, winner_x1, winner_x2, x1, x2, sim_modx, sim_mody, truth_x, truth_y, thresh=0.001):
        xx = sim_modx[winner_x1:winner_x2]
        yy = sim_mody[winner_x1:winner_x2]

        truth_xx = truth_x[x1:x2]
        truth_yy = truth_y[x1:x2]

        end_idx = winner_x2
        if end_idx >= len(sim_modx):
            end_idx = -1
        if sim_modx[winner_x2] < sim_modx[winner_x1]:
            # reverse
            print("Reverse the curve...")
            xx = np.flip(xx)
            yy = np.flip(yy)
        
        end_idx = x2
        if end_idx >= len(truth_x):
            end_idx = -1
        
        if truth_x[end_idx] < truth_x[x1]:
            truth_xx = np.flip(truth_xx)
            truth_yy = np.flip(truth_yy)

        xx_unique, unique_id = np.unique(xx, return_index=True)
        yy_unique = yy[unique_id]
        f = scipy.interpolate.interp1d(xx_unique, yy_unique)

        truth_xx_unique, unique_truth_ind = np.unique(truth_xx, return_index=True)
        truth_yy_unique = truth_yy[unique_truth_ind]
        g = scipy.interpolate.interp1d(truth_xx_unique, truth_yy_unique)

        new_xx = []

        # Find points in both
        x_beg = 0
        x_end = 0
        if (xx[-1] <= truth_xx[0]) or (xx[0] >= truth_xx[-1]):            
            raise Exception(f"This should not happen. Interval does not overlap at all: sim: [{xx[0]}, {xx[-1]}] exp: [{truth_xx[0]}, {truth_xx[-1]}]")
        else:
            x_beg = max(xx[0],  truth_xx[0])
            x_end = min(xx[-1], truth_xx[-1])

        print(f"x_beg={x_beg} and x_end={x_end}")
        interp_common_x = np.linspace(x_beg, x_end, winner_x2 - winner_x1)

        if x_beg > x_end:
            return [], [], []

        for i in range(len(interp_common_x)):
            x = interp_common_x[i]
            n_thresh = thresh*(xx[-1] - xx[0])
            left_bound = xx[0] + n_thresh
            right_bound = xx[-1] - n_thresh
            
            if x > left_bound and x < right_bound:
                new_xx.append(x)
    
        ynew = f(new_xx)
        gnew = g(new_xx)
        
        return new_xx, ynew, gnew


