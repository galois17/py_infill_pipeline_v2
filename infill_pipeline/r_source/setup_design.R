source('r_source/load_libs.R')

options(error=traceback)

# Sets up the design.
# Returns a matrix (not a data frame)
setup_design = function(num_obs, num_of_param) {
  design.init = maximinESE_LHS(lhsDesign(num_obs, num_of_param, seed = 42)$design)$design
  return(design.init)
}

config = read_yaml(Sys.getenv("CONFIG_FILE"))
run_folder = config$system$run_folder
log_r_file = config$system$log_r_file
design_init_file = config$infill$r_design_init_out_file

# Write out the design 
design_init = setup_design(config$infill$design_num_obs, config$infill$design_num_of_param)
design_init_df = as.data.frame(design_init)

write.csv(design_init_df, paste0(run_folder, '/', design_init_file))
flush.console()
write(design_init, stdout())
write("(R subsystem)", file=paste0(run_folder, '/', log_r_file), append=TRUE)
write("Built design init...", file=paste0(run_folder, '/', log_r_file), append=TRUE)
write(dim(design_init_df), file=paste0(run_folder, '/', log_r_file), append=TRUE)



