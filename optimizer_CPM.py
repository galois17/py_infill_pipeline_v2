from abc import ABC, abstractmethod
#from typing import *
from cmath import inf
import traceback
import os
import pandas as pd
import numpy as np
import cpm
import scipy
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


    def get_piecewise(self, sim_modx, sim_mody, truth_x, truth_y, segmentation):
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


