
# Infill Pipeline

This project is a for budgeted multi-objective simulation optimization using a Kriging-based infill algorithm for identification of
crystal plasticity model parameters.


## Setup environment

The application utilizes Python and R. There are packages that must be installed for each:

### Requirements
## Python 3.9
Install Anaconda or Minconda, then setup an environment:
```
conda install --name <env_name> python=3.9
conda activate <env_name>
```

Python packages:

yaml, numpy, pandas, sklearn, pickle5, statsmodels, matplotlib, pyfiglet, pyyaml, joblib, liquidpy, and scipy.

To install packages from requirements file, run:
```
pip install -r requirements.txt
```

### R

R packages that need to be installed:

parallel, ggplot2, dplyr, rsm, desirability, gridExtra, GPareto, DiceDesign, R.matlab, NbClust, R6, purrr, pracma,
readr, pso, pryr, yaml, nat.utils, and openssl.

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
An example (dp780) with the EPSC executables can be found [here](https://universitysystemnh-my.sharepoint.com/:u:/g/personal/kv1033_usnh_edu/Edm_J9RclD9MrzEz0WI8mbwBIH_2S1yPRWPsa1xHXiq6_Q?e=xyyxNu).

### Another example

Only vary the first two parameters for dp780. So, the initial desgin is much smaller (20 observations) than before.

```
system:
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
  budget: 20
  selected_cases: [1,2,3,4,5,6,7,8,9]
  case_ids: [1,2,3,4,5,6,7,8,9]
  alg: 'pso'
  method: 'sms'
  maxit: 20
  #design_num_obs: 150
  design_num_obs: 20
  # Number of params that will be searched in the parameter space,
  # so this must not include fixed parameters. A fixed
  # parameter should be defined in 'lower' and 'upper' with
  # the same value at index j
  design_num_of_param: 2
  design_num_of_responses: 9
  r_design_init_out_file: 'design_init.csv'
  r_design_init_response_out_file: 'design_init_responses.csv'
  # Relative to system.run_folder
  infill_out_file: 'infill_out.csv'
  infill_lock_file: 'infill_out.lock'
  email_notify: 'youremail@localhost.localdomain' 

cpm:
  lower: [1e8, 10, 482782215667.83826, 0.01282147517687075, 652.5227568285271, 623051161.7808537, 880.3136673991768, 669067884567.0271, 0.1852991275615725, 513.4981564026596, 137.77044416754512, 70.0221385738613, 0.002266100185717054, 4.719974896604461]
  upper: [10e8, 100, 482782215667.83826, 0.01282147517687075, 652.5227568285271, 623051161.7808537, 880.3136673991768, 669067884567.0271, 0.1852991275615725, 513.4981564026596, 137.77044416754512, 70.0221385738613, 0.002266100185717054, 4.719974896604461]
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

### Results
The pareto front can be found in /path/to/experiments/folder/pareto_front.csv.

## How to run unit tests
```
coverage run -m pytest **/test_*.py -v
```




