import subprocess
import os
import progressbar
from subprocess import Popen, PIPE, CalledProcessError
import re

import infill_pipeline.config as config
import infill_pipeline.utility as utility

def setup_design():
    """ Setup the initial design"""

    utility.log_r("\nSetting up design matrix\n")
    new_env = os.environ.copy()
    new_env["CONFIG_FILE"] = utility.get_config_file()
    result = subprocess.run(["Rscript", "infill_pipeline/r_source/setup_design.R"], capture_output=True,  text=True, env=new_env)
    print(result.stdout, flush=True)
    print(result.stderr)

def perform_infilling(budget=None, cluster_mode=False):
    """ Perform the infilling by dispatching to R 
    Args: 
        budget: the budget
    """
    regex = re.compile("^(\d+)\s+/")
    max_bar_val = 50
    if budget is not None:
        max_bar_val = budget

    if not cluster_mode:
        bar_infilling = progressbar.ProgressBar(max_value=max_bar_val, 
                                widgets=config.widgets).start()
    new_env = os.environ.copy()
    new_env["CONFIG_FILE"] = utility.get_config_file()
    with Popen(["Rscript", "infill_pipeline/r_source/perform_infilling.R"], stdout=PIPE, bufsize=1, universal_newlines=True, env=new_env) as p:
        for line in p.stdout:
            result = regex.search(line)
            if result:
                print("\n")
                print(line, end='') # process line here
                print("\n")
                budget_num = result.group(1)
                if cluster_mode:
                    print(f"Processing {budget_num}")
                else:
                    bar_infilling.update(int(budget_num))
            else:
                print(line, end='')
                
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)