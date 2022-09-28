import time
import subprocess
import os

class SlurmRunner():
    def __init__(self, path_to_sql_db, config_file):
        self.path_to_sql_db = path_to_sql_db
        self.config_file = config_file

    def dispatch_slurm(self, bb, email):
        job_name = 'infill_ex'

        if bb is None:
            raise Exception("Don't know how to dispatch to slurm...")

        exp = type(bb).__name__.lower()

        if exp == 'CPMDP780'.lower():
            exp = 'dp780'
        elif exp == 'CPMSS316L'.lower():
            exp = 'ss316l'
        else:
            raise Exception("Don't know how to dispatch to slurm...")

        out_file = f"./slurm_output/{job_name}_%A-%a.out"
        err_file = f"./slurm_output/{job_name}_%A-%a.err"
        script_sh = f"""#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH --partition=thrust2
#SBATCH -N 1      # nodes requested
##SBATCH -n 1      # tasks requested
#SBATCH -c 9      # cores requested
#SBATCH --mem=5000  # memory in Mb
#SBATCH -o {out_file}  # send stdout to outfile
#SBATCH -e {err_file}  # send stderr to errfile
#SBATCH -t 0:25:00  # time requested in hour:minute:second
#SBATCH --mail-user={email}
#SBATCH --mail-type=START,END

source ~/.bashrc
conda activate pipeline_gui
cd ~/py_infill_pipeline_client
python run_infiller.py --exp={exp} --sql_db={self.path_to_sql_db} --config_file={self.config_file}
        """

        time_str = time.strftime("%Y%m%d-%H%M%S")
        script_name = f"slurm_infill_dispatch_{time_str}.sh"

        with open(script_name, 'w') as f:
            f.write(script_sh)
            f.flush()

        # Make sure the script gets written out by syncing
        os.sync()

        # Dispatch to the Slurm
        os.system(f"sbatch {script_name}")