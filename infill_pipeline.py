
from xml.dom.expatbuilder import parseFragmentString
from tools.slurm_runner import SlurmRunner
import utility
import r_helper
import os
from data_store import DataStore
from fake_blackbox import FakeBlackbox
import config
import threading
import pandas as pd
import time
import progressbar
import numpy as np
import math
import sys
import traceback
import asyncio
import re
import pickle
import time

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

class InfillPipeline:
    """ InfillPipeline sets up the initial design matrix and runs
    the infilling algorithm.
    """

    # Infill tracker
    infill_count = 0
    infill_count_lock = threading.Lock()

    def __init__(self, run_id, lower, upper, pickle_folder, run_folder,
                 sqlite_file, settings, should_delay_jobs=False,
                 cluster_mode=False,
                 blackbox=None):
        
        self.name = "Infiller"
        self.__run_id = run_id
        self.__pickle_folder = pickle_folder
        self.__design_num_obs = settings['design_num_obs'] or None
        self.__design_num_of_param = settings['design_num_of_param']
        self.__lower = list(map(float, lower))
        self.__upper = list(map(float, upper))
        self.__selected_cases = settings['selected_cases']
        self.__budget = settings['budget']
        self.__run_folder = run_folder
        self.__sqlite_file = sqlite_file
        self.__data_store = DataStore()
        self.__design_csv_file = settings['r_design_init_out_file']
        self.__design_init_response_file = settings['r_design_init_response_out_file']
        self.__infill_out_file = settings['infill_out_file']
        self.__infill_lock_file = settings['infill_lock_file']
        self.__email_notify = settings['email_notify']
        self.__rerun_design_matrix = settings['rerun_design_matrix']
        self.__cluster_mode = cluster_mode
        self.__should_delay_jobs = should_delay_jobs
        self.__pareto_out_file = settings['pareto_out_file'] if 'pareto_out_file' in settings else 'pareto_front.csv'
        if blackbox == None:
            self.__blackbox = FakeBlackbox
        else:
            self.__blackbox = blackbox

        # Track the time it took to process a row
        self.stats = []

    def increment_infill_count(self):
        with InfillPipeline.infill_count_lock:
            InfillPipeline.infill_count += 1

    def set_infill_count(count):
        """ Set the infill count at the class level """
        with InfillPipeline.infill_count_lock:
            InfillPipeline.infill_count = count

    def setup_design(self):
        # Dispatch to helper
        r_helper.setup_design()

    def find_fixed_params(self):
        """ Find fixed params """
        fixed_idx = []
        for j in range(0, len(self.__lower)):
            if self.__lower[j] == self.__upper[j]:
                fixed_idx.append(j)
        return fixed_idx

    def perform_infilling(self):
        # Should we start the BlackBox in a thread?
        # If we dispatch to R, then it should sit in its own thread.
        def monitor_infilling():
            t = threading.currentThread()
            while getattr(t, "do_run", True):
                print(f"\nMonitoring infilling {t.name} {t.ident}\n")
        
                #time.sleep(2)
                lock_file = os.path.join(self.__run_folder, self.__infill_lock_file)
                if not os.path.exists(lock_file):
                    #print("The lock file does not exist...\n")
                    print("\nCheck if infill point can be processed..............\n")
                    time.sleep(4)
                    
                    # Read in the design
                    infill_out_file = os.path.join(self.__run_folder, self.__infill_out_file)
                    if not os.path.exists(infill_out_file):
                        continue

                    df = pd.read_csv(infill_out_file)
                    max_col =  df.shape[1]
                    design = df.iloc[0, 0:(self.__design_num_of_param)].values.tolist()
                    responses = df.iloc[0, self.__design_num_of_param:max_col].values.tolist()

                    if not math.isnan(responses[0]):
                        # Keep waiting...It appears that the R subsystem,
                        # did not get a chance to produce a new file.
                        #
                        # TODO: A more reliable way is to use an md5 check sum                        
                        continue

                    # Call obj function
                    try:
                        print("\nNow able to process the infill point.\n")
                        start = time.time() 
                        # Insert design into the datastore
                        design_as_dict = {}
                        for j in range(0, len(design)):
                            design_as_dict[f"X{j+1}"] = design[j]
                        design_as_dict["infill_type"] = "infill"
                        design_as_dict["infill_id"] = self.__data_store.get_new_infill_id(os.path.join(self.__run_folder, self.__sqlite_file))

                        if type(design_as_dict["infill_id"]) is tuple:
                            design_as_dict["infill_id"] = design_as_dict["infill_id"][0]
                        if isinstance(design_as_dict["infill_id"], str):
                            design_as_dict["infill_id"] = int(design_as_dict["infill_id"])

                        assert isinstance(design_as_dict["infill_id"], int), "The infill point should be an integer"

                        for j in range(0, len(responses)):
                            design_as_dict[f"Y{j+1}"] = None
                        self.__data_store.insert_row(os.path.join(self.__run_folder, self.__sqlite_file), pd.DataFrame(design_as_dict, index=[0]))

                        # Run objective funct
                        print(f"Is clustering on? {self.__cluster_mode}")
                        response = None
                        if self.__cluster_mode:
                            self.__data_store.insert_new_infilling_job(os.path.join(self.__run_folder, self.__sqlite_file), design_as_dict["infill_id"])
                            # design is a a numpy series containing the design (1 row)
                            #full_design = self.join_fixed_values_to_design(design)
                            #full_design = self.rescale_design_to_cpm(full_design)  
                            # 

                            # Launch a slurm task 
                            slurm_runner = SlurmRunner(os.path.join(self.__run_folder, self.__sqlite_file), utility.get_config_file())  
                            slurm_runner.dispatch_slurm(self.__blackbox, self.__email_notify)
                    
                            get_or_create_eventloop().run_until_complete(self.wait_for_objective_funct_execution())

                            # Get the response by looking for response file by infill_d
                            utility.log(f"Finished waiting for {design_as_dict['infill_id']}")
                            pickle_folder = os.path.join(self.__run_folder, self.__pickle_folder)
                            for (dirpath, dirnames, filenames) in os.walk(pickle_folder):
                                for file in filenames:
                                    if re.search(f"_{design_as_dict['infill_id']}\.", file):
                                        # Found the file...
                                        full_path = os.path.join(dirpath, file)
                                        utility.log(f"Found file {full_path} and will try to read the binary file")
                                        f = open(full_path, "rb")
                                        responses_dict = pickle.load(f)
                                        f.close()
                                        
                                        response = self.__blackbox.optim.error_eval_wrap(self.__blackbox.optim.case_fname_as_df, responses_dict)
                                        break
                                        
                        else:
                            response, _ = self.__blackbox.obj_fun(design, False, infill_id=design_as_dict["infill_id"])

                        if response is None:
                            raise Exception("The response should not be null...Not recoverable...")

                        for j in range(0, len(response)):
                            design_as_dict[f"Y{j+1}"] = response[j]

                        c1 = self.__data_store.count(os.path.join(self.__run_folder, self.__sqlite_file))
                        self.__data_store.delete_from_dat_by_infill_id(os.path.join(self.__run_folder, self.__sqlite_file), design_as_dict["infill_id"])
                        c2 = self.__data_store.count(os.path.join(self.__run_folder, self.__sqlite_file))

                        print(f"Total dat is {c1}")
                        print(f"Total dat after deletion is {c2}")
                        assert c2 == c1-1, "Did not delay the row..."

                        self.__data_store.insert_row(os.path.join(self.__run_folder, self.__sqlite_file), pd.DataFrame(design_as_dict, index=[0]))
                        c3 = self.__data_store.count(os.path.join(self.__run_folder, self.__sqlite_file))
                        print(f"Total dat after new row is {c3}")
                        
                        end = time.time() 
                        # Save stat
                        self.stats.append(end-start)
                    except Exception as e:
                        print(traceback.format_exc(),  file=sys.stderr)                        
                        print('The blackbox could not evaulate the obj fun!', file=sys.stderr)
                        
                        utility.log(str(e))
                        #print(e,  file=sys.stderr)
                        raise e
                    # Write back to csv file
                    combined = np.append(design, response)
                    combined = np.reshape(combined, (1, len(combined)))
                    new_df = pd.DataFrame(combined)
                    new_df.to_csv(infill_out_file, index=False)
                    print("Wrote out response for infill point!\n")
                else:
                    print("The lock file exists, so we wait...\n")
                    time.sleep(3)
                
            print("\nTearing down the infill monitoring thread.\n");

        if self.__budget > 0:
            t = threading.Thread(target=monitor_infilling, name='infill_monitor')
            try:
                t.do_run = True
                t.start()   
                # Dispatch to helper
                r_helper.perform_infilling(self.__budget) 
            except Exception as e:
                print('An exception occurred: {}'.format(e))                           
            finally:
                t.do_run = False
                if t.is_alive():
                    t.join()
        else:
            # Budget is <= 0, skip infilling
            return

    async def looper(self):
        total_jobs = self.__data_store.count_all_jobs(os.path.join(self.__run_folder, self.__sqlite_file))
        if total_jobs == 0:
            return

        while True:
            # Check to see if all jobs have been processsed
            count = self.__data_store.count_open_jobs(os.path.join(self.__run_folder, self.__sqlite_file))
            if count > 0:
                # Sleep for a bit more
                print("Sleep for a bit...")
                not_completed = total_jobs -count
                perc = (not_completed/total_jobs)*100
                print(f"Processed {not_completed} out of {total_jobs} or {perc:.2f} %")
                await asyncio.sleep(10)
                print("Woke up and check to see if all jobs are processsed.")
            else:
                print("All done!")
                break

    async def long_operation(self):
        print('long_operation started')
        while True:        
            num_open = self.__data_store.get_open_jobs(os.path.join(self.__run_folder, self.__sqite_file))
            if num_open > 0:
                await asyncio.sleep(3)
        print('long_operation finished')

    async def wait_for_jobs_to_process(self):
        future = asyncio.ensure_future(self.looper())
        #future = asyncio.ensure_future(self.long_operation())
        print('\n\nWaiting for a few seconds')
        await asyncio.sleep(3)
        await future
        print('Done')

    async def wait_for_objective_funct_execution(self):
        await asyncio.sleep(3)
        done = False
        while (not done):
            count = self.__data_store.count_open_infilling_jobs(os.path.join(self.__run_folder, self.__sqlite_file))

            assert (count is not None), "Count should not be None"

            if count >= 0:
                jobs = self.__data_store.get_infilling_jobs(os.path.join(self.__run_folder, self.__sqlite_file))
                
                assert len(jobs) == 1, "There should just be one infilling job"

                infill_id = jobs[0][0]

                t = threading.currentThread()
                print(f"\n\nWaiting for execution to finish for a few seconds: name={t.name}, ident={t.ident}")
                await asyncio.sleep(10)

                # Check if the response file is available now
                print(f"\nLooking for pickle file with infill id of {infill_id}")
                utility.log(f"\nLooking for pickle file with infill id of {infill_id}")
                pickle_folder = os.path.join(self.__run_folder, self.__pickle_folder)
                for (dirpath, dirnames, filenames) in os.walk(pickle_folder):
                    for file in filenames:
                        if re.search(f"_{infill_id}\.", file):
                            full_path = os.path.join(dirpath, file)
                            utility.log(f"\nFound file {full_path}. Stop waiting...")
                            # Found the file...
                            done = True 

    def eval_obj_fun_on_design(self):
        """ 
        Run obj fun against the design matrix that's in
        the datastore.
        """
        design_init_file = os.path.join(
            self.__run_folder, self.__design_csv_file)
        design_init_df = utility.read_entire_csv(design_init_file)
        utility.log('Got pandas dataframe from design csv')
        design_init_df.rename(
            columns={'Unnamed: 0': 'infill_id'}, inplace=True)
        utility.log('The size is ' + str(design_init_df.shape))
        # First row
        utility.log(str(design_init_df.iloc[0]))

        stacked_response = None
        design_init_responses_only_out = os.path.join(self.__run_folder, self.__design_init_response_file)
        if self.__rerun_design_matrix == False and os.path.exists(design_init_responses_only_out):
            # Load the design response 
            loaded_design_init_responses = pd.read_csv(design_init_responses_only_out)
            stacked_response = loaded_design_init_responses.to_numpy()
        else:
            # Make sure indexes pair with number of rows
            design_init_df = design_init_df.reset_index()
            bar_design_df_parse = progressbar.ProgressBar(max_value=design_init_df.shape[0],
                                                        widgets=config.widgets).start()

            for index, row in design_init_df.iterrows():
                self.__data_store.insert_new_job(os.path.join(self.__run_folder, self.__sqlite_file), row['infill_id'])

            for index, row in design_init_df.iterrows():
                # Process each row in the initial design
                start = time.time()                
                self.__data_store.update_job_to_active(os.path.join(self.__run_folder, self.__sqlite_file), row["infill_id"])
                
                if self.__cluster_mode:
                    response, _ = self.__blackbox.obj_fun(row, True, with_metadata=True, infill_id=row["infill_id"], cluster_job_id=f"cluster_{row['infill_id']}")
                else:
                    response, _ = self.__blackbox.obj_fun(row, True, with_metadata=True, infill_id=row["infill_id"], cluster_job_id=None)
                # Done, so mark the job as completed
                self.__data_store.update_job_to_completed(os.path.join(self.__run_folder, self.__sqlite_file), row["infill_id"])
                end = time.time()

                response_as_dict = {}
                for j in range(2, len(row)):
                    # Skip the first 2 since they are metadeta
                    response_as_dict[f"X{j-1}"] = row[j]
                response_as_dict["infill_id"] = row[1]
                
                assert (response is not None), "response must not be null"
                for j in range(0, len(response)):
                    response_as_dict[f"Y{j+1}"] = response[j]
                
                self.__data_store.update_dat(os.path.join(self.__run_folder, self.__sqlite_file), response_as_dict)

                # Save stat
                self.stats.append(end - start)

                print(response)
                bar_design_df_parse.update(index)
                time.sleep(0.1)
                utility.log(f"\nProcessing design matrix at row {index}.\n")
                if stacked_response is None:
                    stacked_response = response
                else:
                    print("\n")
                    print(stacked_response)
                    if hasattr(stacked_response, 'shape'):
                        print(stacked_response.shape)
                    if hasattr(response, 'shape'):  
                        print(response.shape)

                    if isinstance(stacked_response, list):
                        stacked_response = np.array(stacked_response)

                    if isinstance(response, list):
                        response = np.array(response)

                    stacked_response = np.vstack((stacked_response, response))
            
            # TODO: make column names dynamic
            if stacked_response is not None:
                # Write out the response for the design points
                np.savetxt(design_init_responses_only_out, stacked_response,
                        delimiter=",", header="Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9", comments="")

        return stacked_response

    def run(self):
        """ Kick things off. """

        # Setup the design matrix by dispatching to R
        if self.__rerun_design_matrix:
            self.setup_design()

        # Now, load into sqlite3 db
        sqlite_file = os.path.join(self.__run_folder, self.__sqlite_file)
        self.__data_store.print_version(sqlite_file)

        # Create dat/job_queue schema
        if self.__rerun_design_matrix:
            self.__data_store.drop_all(sqlite_file)
            self.__data_store.create_schema(sqlite_file)

            # Remove file
            if os.path.isfile(os.path.join(self.__run_folder, self.__infill_out_file)):
                os.remove(os.path.join(self.__run_folder, self.__infill_out_file))

            # Clear jobs for new run
            self.__data_store.delete_all_from_job_queue(sqlite_file)
            
            design_init_file = os.path.join(
                self.__run_folder, self.__design_csv_file)
            
            # Read design init file
            design_init_df = utility.read_entire_csv(design_init_file)

            # Set the type on the 'infill_type' column
            design_init_df['infill_type'] = [
                config.INFILL_TYPE_DESIGN] * self.__design_num_obs

            # Rename the headers on the pandas df
            utility.rename_design_init_df_inplace(design_init_df)

            # Append columns for responses
            for j in range(1, 20):
                design_init_df[f"Y{j}"] = [None] * self.__design_num_obs

            check_dat = self.__data_store.check_dat_exists(sqlite_file)
            if check_dat:
                # Insert into datastore
                dat_count = self.__data_store.count(sqlite_file)
                if (dat_count > 0):
                    self.__data_store.delete_all(sqlite_file)
            
            # Create the table and insert the initial design
            self.__data_store.insert_design_init(sqlite_file, design_init_df)

        # The action starts here...
        # Evaluate the objective function(s) for each design point
        self.eval_obj_fun_on_design()

        # Finally, start the budgeted sequential infilling process
        self.perform_infilling()

        time.sleep(10)

        # Rescale the pareto front that got generated
        pareto_df = pd.read_csv(os.path.join(self.__run_folder, self.__pareto_out_file))
        
        new_columns = []
        for j in range(len(self.__lower)):
            new_columns.append(f"X{j+1}")
        new_df_of_dict = []
        for _, row in pareto_df.iterrows():
            if type(row) is pd.core.series.Series:
                row_as_list = row.tolist()
            elif type(row) is pd.DataFrame:
                row_as_list = row.values.tolist()
            else:
                raise Exception("Unable to transform row.")

            new_row = self.__blackbox.join_fixed_values_to_design(row_as_list)            
            rescaled_row = self.__blackbox.rescale_design_to_cpm(new_row)
        
            new_dict = {}
            for j in range(len(new_columns)):
                col_name = new_columns[j]
                new_dict[col_name] = rescaled_row[j]

            new_df_of_dict.append(new_dict)

        pareto_df = pd.DataFrame(new_df_of_dict)
        pareto_df.columns = new_columns
        # Write out the rescaled Pareto front
        new_fname = f"final_{self.__pareto_out_file}"
        pareto_df.to_csv(os.path.join(self.__run_folder, new_fname), header=True, index=False)

    def __str__(self):
        return f'InfillPipeline is named {self.name}'

    def __repr__(self):
        return f'InfillPipeline(id={self.__run_id},name={self.name})'
