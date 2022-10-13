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
from optparse import OptionParser
from simple import *
from optimizer_simple_case import OptimizerSimpleCase
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import interpolate
#from matplotlib.mlab import griddata
from matplotlib import cm
from scipy.interpolate import griddata

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

def plot():
    lower = config.data_loaded['simple']['lower']
    upper = config.data_loaded['simple']['upper']

    path_to_db = os.path.join(config.data_loaded['system']['run_folder'], config.data_loaded['system']['sqlite_db'])
    print(path_to_db)

    path_to_pareto = os.path.join(config.data_loaded['system']['run_folder'], 'pareto_front.csv')
    df = pd.read_csv(path_to_pareto)
    p_x1 = df.iloc[0,0]
    p_x2 = df.iloc[0,1]
    p_y = df.iloc[0,2]

    p_x1 = utility.rescale_val(p_x1, 0, 1, lower[0], upper[0])
    p_x2 = utility.rescale_val(p_x2, 0, 1, lower[0], upper[0])

    ds = DataStore()
    df = ds.get_all_infill(path_to_db)
    print(df)
    x1 = df['X1'].to_numpy()
    print(x1)
    x2 = df['X2'].to_numpy()
    y1 = df['Y1'].to_numpy()

    x1 = utility.rescale_val(x1, 0, 1, lower[0], upper[0])
    x2 = utility.rescale_val(x2, 0, 1, lower[1], upper[1])

    y1 = y1.astype(np.float64)
    print(y1)

    model_x_data = np.linspace(min(x1), max(x1), 30)
    model_y_data = np.linspace(min(x2), max(x2), 30)

    XX, YY = np.meshgrid(model_x_data, model_y_data)
    Z = griddata((x1, x2), y1, (XX, YY),  method='cubic')

    fig1 = plt.figure(1)
    ax= plt.axes(projection='3d')

    #f = interpolate.interp2d(model_x_data, model_y_data, z, kind='cubic')
    #surf = ax.plot_surface(XX, YY, Z, rstride=5, cstride=5, cmap=cm.jet,
    #                 linewidth=1, antialiased=True)
    surf = ax.plot_surface(XX, YY, Z, rstride=5, cstride=5,
                    linewidth=1, antialiased=True)
    ax.scatter(x1, x2, y1, color='r')

    sz = 80
    ax.scatter(p_x1, p_x2, p_y, s=sz, color='g', label="optimal")
    ax.legend()
    plt.show()

    #############################################################

    fig2 = plt.figure(2)
    ax2 = plt.axes()
    ax2.contourf(XX, YY, Z)
    
    ax2.scatter(x1, x2, s=20, color='r')
    ax2.scatter(p_x1, p_x2, s=sz, color='g', label="optimal")
    ax2.legend()
    plt.show()    

def main():
    config.parser.add_option("--plot", action="store_true", dest="plot", help="plot the data from a previous run and exit")

    (config.options, config.args) =  config.parser.parse_args()
    if config.options.show_welcome:
        utility.show_welcome()
    
    if config.options.version:
        print(f"\n\n\nVersion {config.__VERSION__}")
        return None

    config_file = os.environ['CONFIG_FILE']
    with open(config_file, 'r') as stream:
        config.data_loaded = yaml.safe_load(stream)

    # Plot and then exit
    if config.options.plot:
        plot()
        return None

    cur_datetime = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    # Save the current datetime
    config.cur_datetime = cur_datetime

    run_id = utility.generate_run_id(cur_datetime)

    # Instance of a Blackbox
    bb = SimpleCase(run_id, config.data_loaded['system']['run_folder'], config.data_loaded['simple']['lower'], config.data_loaded['simple']['upper'], config.data_loaded['simple'])
    print(bb.__repr__())

    # Set everything up
    if not config.data_loaded['simple']['should_rerun_setup']:
        utility.log("!!Do not run setup again!!!\n\n")
    else:
        utility.log("!!Re-run setup!!!\n\n")

    # Instance of the Pipeline
    ip = InfillPipeline(run_id, 
        config.data_loaded['simple']['lower'],
        config.data_loaded['simple']['upper'],
        config.data_loaded['simple']['pickle_folder'],
        config.data_loaded['system']['run_folder'],
        config.data_loaded['system']['sqlite_db'],
        config.data_loaded['infill'],
        should_delay_jobs=config.data_loaded['system']['should_delay_jobs'],
        cluster_mode=config.data_loaded['system']['cluster_mode'],
        blackbox=bb
        )
    print(ip.__repr__())

    if PRINT_CONFIG:
        print(config.data_loaded['system'])
        print(config.data_loaded['infill'])
        print(config.data_loaded['simple'])
        selected_cases = config.data_loaded['infill']['selected_cases']

        lower = config.data_loaded['simple']['lower']
        new_lower = [float(x) for x in lower] 
        print(new_lower)  

    run_folder =  config.data_loaded['system']['run_folder']
    
    # Check to see if run folder exists
    if not os.path.isdir(run_folder):
        raise InfillError("Failure: Folder %s does not exist..." % (run_folder))
    
    sqlite_db = config.data_loaded['system']['sqlite_db']
    if not os.path.exists(os.path.join(run_folder, sqlite_db)):
        print(f"Warning: File {os.path.join(run_folder, sqlite_db)} does not exist... It will get created.")
    
    t = threading.Thread(target=monitor_ext)
    try:
        t.do_run = True
        t.start()  
        ip.run()                              

    finally:
        utility.log_run()
        t.do_run = False
        t.join()

    # Report stats
    avg_time_in_min = statistics.mean(ip.stats)/60 
    min_time_in_min = min(ip.stats)/60
    max_time_in_min = max(ip.stats)/60
    print("\n==============================================\n")
    print(f"\nStat of processing time for each data point: \n  avg={avg_time_in_min:.3f}m min={min_time_in_min:.3f}m max={max_time_in_min:.3f}m\n\n")
    print("\n==============================================\n")

    utility.log_run()

if __name__ == "__main__":
    main()