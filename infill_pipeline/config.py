from optparse import OptionParser
import time
import os
import progressbar
from optparse import OptionParser

__VERSION__ = '0.0.1'

data_loaded = None
cur_datetime = None

INFILL_TYPE_DESIGN = "design"
INFILL_TYPE_INFILLED = "infilled"

widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]
progress_bar = None

parser = OptionParser()
parser.add_option("--debug", action="store_true", dest="debug", help="debug")
parser.add_option("--show_welcome", action="store_true", dest="show_welcome", help="show_welcome")
parser.add_option("--verbose", action="store_true", dest="verbose", help="verbose")
parser.add_option("--version", action="store_true", dest="version", help="version")
parser.add_option("--run_batch", action="store", type="string", dest="run_batch", help="run a simulator against a batch of points (no infilling)")

(options, args) = (None, None)

class ConfigInvalid(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message
        
    def __str__(self):
        return self.message

def check_config(config_data):
    """ Check config
    Args:
        config_data: the config data
    Return:
    """
    run_folder = config_data['system']['run_folder']
    if not os.path.exists(run_folder):
        raise ConfigInvalid(f"'system'.'run_folder'->{run_folder} does not exist")
    
    return True