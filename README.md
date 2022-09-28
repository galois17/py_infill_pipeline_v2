
# Infill Pipeline

This project is a for budgeted multi-objective simulation optimization using a Kriging-based infill algorithm for identification of
crystal plasticity model parameters.


## Setup environment

### Requirements
## Python 3.9
```
conda install --name <env_name> python=3.9
conda activate <env_name>
```

Python packages:

yaml, numpy, pandas, sklearn, pickle5, statsmodels, matplotlib, pyfiglet, pyyaml, joblib, liquidpy, scipy

### R

R packages:

parallel, ggplot2, dplyr, rsm, desirability, gridExtra, GPareto, DiceDesign, R.matlab, NbClust, R6, purrr, pracma,
readr, pso, pryr, yaml, nat.utils, openssl

## How to run

```
CONFIG_FILE=/path/to/config.yaml python runner_DP780.py
```

Sample config.yaml file:
```
system:
  dry_run: False
  run_folder: '/path/to/experiments/folder'
  log_r_file: 'log_r.txt'
  sqlite_db: 'sqlite.db'
  # Timeout in seconds
  subprocess_timeout: 1200
  # Delay jobs for processing of initial design points.
  should_delay_jobs: False
  cluster_mode: False
  n_jobs: 9

infill:
  rerun_design_matrix: True
  budget: 140
  selected_cases: [1,2,3,4,5,6,7,8,9]
  case_ids: [1,2,3,4,5,6,7,8,9]
  alg: 'pso'
  method: 'sms'
  maxit: 20
  design_num_obs: 140
  # Number of params that will be searched in the parameter space,
  # so this must not include fixed parameters. A fixed
  # parameter should be defined in 'lower' and 'upper' with
  # the same value at index j
  design_num_of_param: 14
  design_num_of_responses: 9
  r_design_init_out_file: 'design_init.csv'
  r_design_init_response_out_file: 'design_init_responses.csv'
  # Relative to system.run_folder
  infill_out_file: 'infill_out.csv'
  infill_lock_file: 'infill_out.lock'
  email_notify: 'youremail@localhost.localdomain'

cpm:
  lower: [1e8, 10, 1e10, 1e-3, 1e2, 1e8,       10, 1e10,1e-2,1e2, 50,50,1e-3, 1]
  upper: [10e8, 100, 100e10, 20e-3,10e2, 10e8,100e1,100e10,20e-2, 10e2, 150, 150,10e-3,5]
  param_fname: ['ferDD.sx','martDD.sx']
  param_fname_template: ['ferDD.sx.liquid', 'martDD.sx.liquid']
  # Relative to system.run_folder
  EPSC_source_folder: 'EPSCSource'
  # Relative to system.run_folder
  running_folder: 'RunningFolder'
  should_rerun_setup: True
  # Assume that the executable produces this output file
  EPSC_output_file: 'epsc3.out'
  EPSC_linux_executable: 'a.out'
  # Relative to system.run_folder
  case_fname: ['InputFiles', 'Inp_WhatToFit_DP.csv']
  # Relative to system.run_folder
  fit_param_fname: ['InputFiles', 'Inp_FittingParams_DP.csv']
  recipe_fname: ['InputFiles', 'Inp_Fit_recipes_DP.in']
  info_fname: ['InputFiles', 'Inp_Info_DP.yaml']
  pickle_folder: 'out'
```
An example can be found [here](https://universitysystemnh-my.sharepoint.com/:u:/g/personal/kv1033_usnh_edu/Edm_J9RclD9MrzEz0WI8mbwBIH_2S1yPRWPsa1xHXiq6_Q?e=xyyxNu) 
## How to run unit tests
```
python -m unittest discover -v
```
or
```
pytest test_*.py -v
```

## Create docs

From docs, run:
```
make html
```


