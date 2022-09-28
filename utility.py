from multiprocessing.util import is_exiting
import numpy as np
import config
import os
import json
import datetime
import pandas as pd 
import infill_pipeline
from pyfiglet import Figlet
import time
import random
from pathlib import Path

# Generate a RUN_ID to identify this run
def generate_run_id(cur_datetime):
    """ Generate a run id """
    id = "ID_" + cur_datetime
    return id

def get_datetime():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def get_config_file():
    """ Get config yaml file """
    config_file = os.environ['CONFIG_FILE']
    if not config_file:
        f = "config.yaml"
        path = Path(f)
        if path.is_file():
            return f
        else:
            raise RuntimeError(f"The config file {f} does not exist.")
    else:
        path = Path(config_file)
        if path.is_file():
            return config_file
        else:
            raise RuntimeError(f"The config file {config_file} does not exist.")

def show_welcome():
    fonts = ['rounded', 'script', 'hollywood']
    j = random.randint(0, len(fonts)-1)
    custom_fig = Figlet(font=fonts[j])
    print(custom_fig.renderText('Welcome!!'))
    time.sleep(2)

def log(message):
    run_folder =  config.data_loaded['system']['run_folder']
    with open(os.path.join(run_folder, 'log.txt'), 'a+') as f:
        f.write(datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
        f.write("\n")
        f.write(message)
        f.write("\n")
        f.flush()

def get_n_jobs():
    n_jobs = config.data_loaded['system']['n_jobs']
    if n_jobs is None:
        n_jobs = -1
    return n_jobs

def log_run():
    run_folder =  config.data_loaded['system']['run_folder']
    with open(os.path.join(run_folder, 'log_run.txt'), 'w') as f:
        f.write("Started at " + config.cur_datetime + "\n\n")
        f.write(json.dumps(config.data_loaded))
        f.write("\n\n")
        f.write(datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
        f.write("\n")
        f.flush()

def log_r(message):
    run_folder =  config.data_loaded['system']['run_folder']
    with open(os.path.join(run_folder, 'log_r.txt'), 'a+') as f:
        f.write(datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
        f.write("\n")
        f.write(message)
        f.write("\n")
        f.flush()

def read_entire_csv(filepath):
    """ Return the csv data as a Pandads dataframe
    """
    in_memory_file = None
    #with open(filepath, 'r') as file:
    #    in_memory_file = file.read()
    data = pd.read_csv(filepath)
    return data

def rename_design_init_df_inplace(df):
    df.rename(columns={'Unnamed: 0': 'infill_id', 'V1': 'X1', 'V2': 'X2', 'V3': 'X3', 'V4': 'X4',
        'V5': 'X5', 'V6': 'X6', 'V7': 'X7', 'V8': 'X8','V9': 'X9',
        'V10': 'X10','V11': 'X11', 'V12': 'X12', 'V13': 'X13',
        'V14': 'X14', 'V15': 'X15'
    }, 
    inplace=True)

    infill_ids = df['infill_id']
    infill_pipeline.InfillPipeline.set_infill_count(max(infill_ids))

def fixed_bound(lower, upper):
    """ Get fixed values in the bounds """
    idx = []
    for j in range(0, len(lower)):
        if lower[j] == upper[j]:
            idx.append(j)
    return idx

def rescale_val(val, min_s, max_s, min_t, max_t):
    """ Rescale a value from given range to target range """
    ret = (val - min_s)*(max_t - min_t)/(max_s - min_s) + min_t
    return ret


def rescale_vec(vec, min_t, max_t):
    """ Rescale vector where source min and max are obtained from the vector """
    min_s = [min(vec)]*len(vec)
    max_s = [max(vec)]*len(vec)
    min_t = [min_t]*len(vec)
    max_t = [max_t]*len(vec)
    ret = rescale_val(np.array(vec), np.array(min_s), np.array(max_s), np.array(min_t), np.array(max_t))
    return ret.tolist()

def rescale_vec_with_source(vec, min_s, max_s, min_t, max_t):
    ret = rescale_val(np.array(vec), np.array(min_s), np.array(max_s), np.array(min_t), np.array(max_t))
    return ret.tolist()

def smooth(a, WSZ):
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def smooth2(x, window_len=11, window='hanning'):
    """Smooth the data using a window with requested size.

    FROM: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


