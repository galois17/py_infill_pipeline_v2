import threading
import logging
import progressbar
import time
import datetime
import yaml
from select import select
import os
import sys
import pandas as pd

from infill_pipeline.impl.optimizer_DP780 import OptimizerDP780
from infill_pipeline.impl.cpm_DP780 import CPMDP780
import infill_pipeline.utility as utility
from infill_pipeline.infill_error import InfillError
import infill_pipeline.config as config
from infill_pipeline.infill_pipeline import InfillPipeline

progressbar.streams.wrap_stderr()
logging.basicConfig()

PRINT_CONFIG = True

config.progress_bar = progressbar.ProgressBar(max_value=300,
                                              widgets=config.widgets).start()

def monitor_ext():
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        print("\nMonitoring EPSC app...\n")
        time.sleep(60)
    print("\nCleanup and and exit\n")

def main():
    print("Start")
    (config.options, config.args) = config.parser.parse_args()
    if config.options.show_welcome:
        utility.show_welcome()

    if config.options.version:
        print(f"\n\n\nVersion {config.__VERSION__}")
        return None

    cur_datetime = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    # Save the current datetime
    config.cur_datetime = cur_datetime

    # Generate a run id
    run_id = utility.generate_run_id(cur_datetime)

    config_file = utility.get_config_file()
    with open(config_file, 'r') as stream:
        config.data_loaded = yaml.safe_load(stream)

    # Instance of a Blackbox
    optim = OptimizerDP780(config.data_loaded['system']['run_folder'],
                           config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname'])
    bb = CPMDP780(run_id, config.data_loaded['system']['run_folder'],
                  config.data_loaded['infill']['r_design_init_out_file'],
                  config.data_loaded['infill']['design_num_obs'],
                  config.data_loaded['infill']['selected_cases'],
                  config.data_loaded['infill']['case_ids'],
                  config.data_loaded['system']['subprocess_timeout'],
                  config.data_loaded['cpm'],
                  optim
                  )
    print(bb.__repr__())
    # Set everything up
    if not config.data_loaded['cpm']['should_rerun_setup']:
        utility.log("!!Do not run setup again!!!\n\n")
    else:
        utility.log("!!Re-run setup!!!\n\n")

    bb.setup(should_rerun_epsc_copy=config.data_loaded['cpm']['should_rerun_setup'],
            should_refit_exp_data=False)

    # Instance of the Pipeline
    ip = InfillPipeline(run_id,
        config.data_loaded['cpm']['lower'],
        config.data_loaded['cpm']['upper'],
        config.data_loaded['cpm']['pickle_folder'],
        config.data_loaded['system']['run_folder'],
        config.data_loaded['system']['sqlite_db'],
        config.data_loaded['infill'],
        should_delay_jobs=config.data_loaded['system']['should_delay_jobs'],
        cluster_mode=config.data_loaded['system']['cluster_mode'],
        blackbox=bb
        )
    if PRINT_CONFIG:
        print(config.data_loaded['system'])
        print(config.data_loaded['infill'])
        print(config.data_loaded['cpm'])

        lower = config.data_loaded['cpm']['lower']
        new_lower = [float(x) for x in lower] 
        print(new_lower)  

    run_folder =  config.data_loaded['system']['run_folder']
    
    # Check to see if run folder exists
    if not os.path.isdir(run_folder):
        raise InfillError("Failure: Folder %s does not exist..." % (run_folder))

    t = threading.Thread(target=monitor_ext)
    try:
        t.do_run = True
        t.start()  
        if not config.options.run_batch:
            ip.run()  
        else:            
            print(f"Run batch...{config.options.run_batch}")
            path_to_batch_csv = os.path.join(run_folder, config.options.run_batch)
            batch_pd = pd.read_csv(path_to_batch_csv)

            for index, row in batch_pd.iterrows():
                response,_ = bb.obj_fun(row, rescale=False)
                print(response)
                
            print("Cleaning up afer ourselves....")
            time.sleep(10)
        t.do_run = False
        if t.is_alive():
            t.join()
    except (KeyboardInterrupt, SystemExit):
        print('\n!Received keyboard interrupt, quitting threads.\n')
        t.do_run = False
        if t.is_alive():
            t.join()                         
    finally:
        utility.log_run()

if __name__ == '__main__':
    main()
