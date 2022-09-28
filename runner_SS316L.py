from select import select
import yaml
from infill_pipeline import *
from optparse import OptionParser
import config
import os
from infill_error import InfillError
import utility
import datetime
from impl.cpm_SS316L import *
import time
import progressbar
import logging
import threading
import statistics
from impl.optimizer_SS316L import OptimizerSS316L

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
    print("\nCleanup and and exit\n");

def main():
    (config.options, config.args) =  config.parser.parse_args()
    if config.options.show_welcome:
        utility.show_welcome()
    
    if config.options.version:
        print(f"\n\n\nVersion {config.__VERSION__}")
        return None

    cur_datetime = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    # Save the current datetime
    config.cur_datetime = cur_datetime

    run_id = utility.generate_run_id(cur_datetime)

    config_file = os.environ['CONFIG_FILE']
    with open(config_file, 'r') as stream:
        config.data_loaded = yaml.safe_load(stream)

    # Instance of a Blackbox
    optim = OptimizerSS316L(config.data_loaded['system']['run_folder'], 
        config.data_loaded['cpm']['case_fname'], config.data_loaded['cpm']['fit_param_fname'], config.data_loaded['cpm']['recipe_fname'], config.data_loaded['cpm']['info_fname'])
    bb = CPMSS316L(run_id, config.data_loaded['system']['run_folder'],       
        config.data_loaded['infill']['r_design_init_out_file'],
        config.data_loaded['infill']['design_num_obs'],
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

    if not config.options.run_batch:
        # Report stats
        avg_time_in_min = statistics.mean(ip.stats)/60 
        min_time_in_min = min(ip.stats)/60
        max_time_in_min = max(ip.stats)/60
        print("\n==============================================\n")
        print(f"\nStat of processing time for each data point: \n  avg={avg_time_in_min:.3f}m min={min_time_in_min:.3f}m max={max_time_in_min:.3f}m\n\n")
        print("\n==============================================\n")

if __name__ == "__main__":
    main()
