
from abc import ABC, abstractmethod
#from this import d
#from typing import *
import numpy as np
import re
import os
import pandas as pd
import shutil
import copy
from liquid import Liquid
import traceback
from joblib import Parallel, delayed
import pickle
import datetime
import platform
import signal
import subprocess
import sys
import random

from blackbox import BlackBox
import utility
import config
import liquid_template_helper

class CPM(BlackBox):
    def __init__(self, run_id, run_folder, design_csv_file, design_number_of_obs, selected_cases, case_ids, subprocess_timeout, settings, optim):
        super().__init__()
        self.name = "CPM"
        self._run_id = run_id
        self._param_fname = settings['param_fname']
        self._param_fname_template = settings['param_fname_template']
        self._lower = list(map(float, settings['lower']))
        self._upper = list(map(float, settings['upper']))
        self._run_folder = run_folder
        self._EPSC_source_folder = settings['EPSC_source_folder']
        self._EPSC_linux_executable = settings['EPSC_linux_executable']
        self._EPSC_output_file = settings['EPSC_output_file']
        self._design_csv_file = design_csv_file
        self._design_number_of_obs = design_number_of_obs
        self._selected_cases = selected_cases
        self._case_ids = case_ids
        self._running_folder = settings['running_folder']
        self._case_fname = settings['case_fname']
        self._fit_param_fname = settings['fit_param_fname']
        self._recipe_fname = settings['recipe_fname']
        self._info_fname = settings['info_fname']
        self._pickle_folder = settings['pickle_folder']
        self._subprocess_timeout = subprocess_timeout
        self._infill_run_folder_name = 'infill'
        self.optim = optim

    def setup(self, should_rerun_epsc_copy=True, should_refit_exp_data=True):
        """ Setup the environment for the experiment.
        """
        print("Setup optimizer!")
        self.optim.setup()
        # Prep exp data
        if should_refit_exp_data:
            self.optim.setup_exp_data()
        print("Finished setup of optimizer!")

        if not should_rerun_epsc_copy:
            return None
        
        folder = os.path.join(self._run_folder, self._EPSC_source_folder)
        if not os.path.isdir(folder):
            raise Exception('EPSC folder does not exist')

        row_count = self._design_number_of_obs
        
        # Copy contents from EPSC folder
        lowest_valid_case = self._selected_cases[0]

        for j in range(0, len(self._selected_cases)):
            case_id = self._selected_cases[j]
            
            copy_source = os.path.join(
                self._run_folder, self._EPSC_source_folder, str(case_id))
            copy_dest = os.path.join(
                self._run_folder, self._running_folder, str(lowest_valid_case))
            copy_dest_deep = os.path.join(
                self._run_folder, self._running_folder, str(lowest_valid_case), str(case_id))
            
            if not os.path.exists(copy_dest):
                os.makedirs(copy_dest, exist_ok=True)
            if not os.path.exists(copy_dest_deep):
                os.makedirs(copy_dest_deep, exist_ok=True)
            shutil.copytree(copy_source, copy_dest_deep, dirs_exist_ok=True)
    
        def copy_runs(j):
            """ Make copies of the first design point """
            copy_source = os.path.join(
                self._run_folder, self._running_folder, str(1))
            copy_dest = os.path.join(
                self._run_folder, self._running_folder, str(j))
            shutil.copytree(copy_source, copy_dest, dirs_exist_ok=True)

        # Copy the folder in parallel for speed
        results = Parallel(
            n_jobs=utility.get_n_jobs(), backend="threading")(map(delayed(copy_runs), range(2, row_count+1)))

    def write_dps_xfile(self, template_file, args):
        """
        Write out dps xfile with the parameters
        Args:
            template_file: the liquid template
            design: the design vector       
        Returns: 
            evaluated templated
        Raises:
        """
        
        try:
            liq = Liquid(template_file,  filters={'filter_string_format': liquid_template_helper.filter_string_format})
            ret = liq.render({'par': args})
        except Exception as e:
            print("This should not happen with the templates..." + str(e))
            print(traceback.format_exc())
            raise e
        return ret

    def obj_fun(self, design, is_initial_design_point, with_metadata=False, infill_id=None, rescale=True, num_jobs=-1, cluster_job_id=None):
        """ Objective function
        Args:
            design: A numpy series containing the design (1 row). If the values are not
            between 0 and 1, set rescale=False
        Returns:
            error: the error as an np array
            response: additional details
        """
        utility.log("(CPM) processing a row: ")
        if isinstance(design, pd.DataFrame):
            utility.log("This is a dataframe")
            utility.log(str(design.shape))
            design = design.values.tolist()
            design = design[0]
            utility.log(str(design))
        elif not isinstance(design, list):
            print(design.to_string())
            utility.log(str(type(design)))
            utility.log(design.to_string())

            design = design.tolist()
        else:
            utility.log(str(type(design)))
            utility.log(str(design))

        if with_metadata:
            # Index and infill_id at loc 0 and 1            
            design = design[2:len(design)]
        
        # Add fixed points to the design
        full_design = self.join_fixed_values_to_design(design)

        if rescale:
            full_design = self.rescale_design_to_cpm(full_design)

        infill_id_str = ""
        if infill_id:
            infill_id_str = str(int(infill_id))

        # Write template to the proper file location
        base_folder = None
        for j in range(0, len(self._selected_cases)):
            case_id = self._selected_cases[j]
            folder = None    
            if cluster_job_id is None:
                if not is_initial_design_point:
                    base_folder = os.path.join(self._run_folder, self._running_folder, self._infill_run_folder_name)
                    folder = os.path.join(base_folder, str(case_id))
                else:
                    base_folder = os.path.join(self._run_folder, self._running_folder, infill_id_str)
                    folder = os.path.join(base_folder, str(case_id))
                # Copy the EPSC source to the folder
                copy_source = os.path.join(
                    self._run_folder, self._EPSC_source_folder, str(case_id))
            else:
                base_folder = os.path.join(self._run_folder, self._running_folder, cluster_job_id) 
                folder = os.path.join(base_folder, str(case_id))
                # Copy the EPSC source to the folder
                copy_source = os.path.join(
                    self._run_folder, self._EPSC_source_folder, str(case_id))
        
            # Create 'infill' or custom cluster_job_id folder to run EPSC apps in
            # TODO: clean this up
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            shutil.copytree(copy_source, folder, dirs_exist_ok=True)

        #####################
        num_responses = len(self._case_ids)

        # Collect the valid case folders
        collect_valid_folders = []
        for name in os.listdir(base_folder):
            if not name.startswith('.'):
                # Drill into the RunningFolder/, RunningFolder/infill or RunningFolder/{cluster_job_id}
                full_path = None
                if cluster_job_id is None:
                    if not is_initial_design_point:
                        full_path = os.path.join(self._run_folder, self._running_folder, self._infill_run_folder_name, name)
                    else:
                        full_path = os.path.join(self._run_folder, self._running_folder, infill_id_str, name)
                else:
                    full_path = os.path.join(self._run_folder, self._running_folder, cluster_job_id, name)
                if os.path.isdir(full_path):
                    collect_valid_folders.append(name)
        
        # Parallelize the execution of the EPSC apps in each case
        if cluster_job_id:
            param_for_parallel = map(lambda x: (full_design, x, cluster_job_id), collect_valid_folders)
        else:
            if not is_initial_design_point:
                loc = self._infill_run_folder_name
            else:
                loc = infill_id_str
            param_for_parallel = map(lambda x: (full_design, x, loc), collect_valid_folders)

        results = Parallel(n_jobs=utility.get_n_jobs())(map(delayed(self.write_out_dps_xfile_and_execute), param_for_parallel))               

        # Pull out the results
        responses_dict = {}
        for v in results:
            case_folder = v[0]
            data = v[1]
            responses_dict[case_folder] = data

        # Pickle the data (full design and response)
        cur_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        pickle_name = f"response_{infill_id_str}.dat.{cur_date}"

        full_pickle_folder_path = os.path.join(self._run_folder, self._pickle_folder)
        if not os.path.isdir(full_pickle_folder_path):
            os.mkdir(full_pickle_folder_path)
        with open(os.path.join(self._run_folder,  self._pickle_folder, pickle_name), 'wb') as handle:
            pickle.dump(responses_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        response = self.optim.error_eval_wrap(self.optim.case_fname_as_df, responses_dict)
        utility.log(str(response))
        return response, responses_dict

    def write_out_dps_xfile_and_execute(self, param):
        """ Write out the DPS xFiles and then execute the EPSC app 
        Args:
            param: tuple (design, case_folder)
        Returns:

        """
        full_design = param[0]
        case_folder = param[1]        
        if param[2]:
            name = param[2]
        else:
            name = ""

        # Write to each folder
        for j in range(0, len(self._param_fname_template)):
            t = self._param_fname_template[j]
            # Get the liquid template file
            template = os.path.join(self._run_folder, t)
            file_name_out = self._param_fname[j]
            ret = self.write_dps_xfile(template, full_design)
            # Copy EPSC source to "infill" folder
            target = os.path.join(self._run_folder, self._running_folder, name, case_folder, file_name_out)
            print(f"Wrote template {template} to {target}")
            with open(target, 'w') as file:
                # Write out the liquid template for the EPSC app
                file.write(ret)
        # Run the EPSC executable in the target folder
        if platform.system() == 'Linux':
            infill_running_folder = os.path.join(self._run_folder, self._running_folder, name, case_folder)
            print(f"\nProcessing case {case_folder}\n")
            print(f"Infill running folder is {infill_running_folder}\n")
            print(f"Started the EPSC app...")

            try:
                p = subprocess.Popen(['./' + self._EPSC_linux_executable], cwd=infill_running_folder, stdout=subprocess.DEVNULL, start_new_session=True)
                p.wait(timeout=self._subprocess_timeout)

                print(f"Finished the EPSC app...")
                vpsc_out_file = os.path.join(self._run_folder, self._running_folder, name, case_folder, self._EPSC_output_file)
                vpsc_out = CPM.read_from_vps_cout(vpsc_out_file, skip_no=6)
                subset_vpsc_out = vpsc_out[:, [ 0, 6, (vpsc_out.shape[1] - 8) ]]

                return (case_folder, subset_vpsc_out)
            except subprocess.TimeoutExpired:
                print(f'EPSC app timed out after {self._subprocess_timeout}s', file=sys.stderr)
                print('Terminating the process...', file=sys.stderr)
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                vpsc_out_file = os.path.join(self._run_folder, self._running_folder, name, case_folder, self._EPSC_output_file)
                vpsc_out = CPM.read_from_vps_cout(vpsc_out_file, skip_no=6)
                subset_vpsc_out = vpsc_out[:, [ 0, 6, (vpsc_out.shape[1] - 8) ]]
                
                return (case_folder, subset_vpsc_out)

        elif platform.system() == 'Windows':
            raise NotImplementedError
        else:
            print("This platform is not supported...")
            # Read in the results
            samples_loc = os.path.join(self._run_folder, 'output_collection')
            if not os.path.exists(samples_loc):
                raise RuntimeError(f"The folder {samples_loc} does not exist. It's needed for testing when the platform is not supported.")
            sample_out = os.listdir(samples_loc)
            n = random.randint(0, len(sample_out)-1)
            print(f"\nProcessing case {case_folder}\n")
            vpsc_out = CPM.read_from_vps_cout(os.path.join(samples_loc, sample_out[n])  )
            subset_vpsc_out = vpsc_out[:, [ 0, 6, (vpsc_out.shape[1] - 8) ]]
            print(f"\nFinished processing case {case_folder}\n")
            return (case_folder, subset_vpsc_out)

    @staticmethod
    def read_from_vps_cout(path_to_out, skip_no=0):
        """
        Read results.

        Args:
            path_to_out: path to vps cout     
        Returns: 
            A numpy matrix
        Raises:
        """
        data = None
        with open(path_to_out) as fp:
            for _ in range(skip_no):
                next(fp)

            line = fp.readline()
            while line:
                results = re.split("\s+", line.lstrip())
                filtered = list(filter(None, results))

                res = list(map(float, filtered))
                res_np = np.array(res, dtype=np.float32)
                if data is None:
                    data = res_np
                else:
                    data = np.vstack((data, res_np))
                line = fp.readline()
        return data