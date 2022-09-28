source('r_source/load_libs.R')

options(error=traceback)

config = read_yaml(Sys.getenv("CONFIG_FILE"))
run_folder = config$system$run_folder
log_r_file = config$system$log_r_file

NUM_PARAMS = config$infill$design_num_of_param
NUM_CASES = config$infill$design_num_of_responses
BUDGET = config$infill$budget
# Options are pso, genoud, or random
ALG = "pso"
# Options are SMS or EHI
METHOD = "SMS"
MAXIT = 20

R_REF_POINT = rep(0.1, NUM_CASES)

NUM_PARAMS = config$infill$design_num_of_param
INFILL_OUT_FILE = config$infill$infill_out_file

#MAX_WAIT_ROUND = 200
MAX_WAIT_ROUND = 100
LOCK_FILE = config$infill$infill_lock_file

mesg = sprintf("config_file is %s, number_of_params %s", config, NUM_PARAMS) 
write(mesg, file=paste0(run_folder, '/', log_r_file), append=TRUE)

obj_fun = function(x) {
    # Dispatches to actual objective fun.
    # An INFILL_OUT_FILE is created with the infill design point.
    # It's considered processed once the file no longer esists.
    # It will wait poll to see if the file still exists...
    #
    # Args:
    #   x: design point as numerical vector
    on.exit(closeAllConnections())

    write('Eval obj function: ', file=paste0(run_folder, '/', log_r_file), append=TRUE)
    #write(as.character(t(x)), file=paste0(run_folder, '/', log_r_file), append=TRUE)
    write(as.character(x), file=paste0(run_folder, '/', log_r_file), append=TRUE)
    write(class(x), file=paste0(run_folder, '/', log_r_file), append=TRUE)
    write(INFILL_OUT_FILE, file=paste0(run_folder, '/', log_r_file), append=TRUE)

    # Write out infill point to csv
    out_infill_point = c(x, rep(NA, NUM_CASES))
    touch(paste0(run_folder, '/', LOCK_FILE))
    write.csv(t(out_infill_point), file=paste0(run_folder, '/', INFILL_OUT_FILE), row.names=FALSE)
    write.csv(t(out_infill_point), file=paste0(run_folder, '/', 'infill_out2.csv'), row.names=FALSE)
    unlink(paste0(run_folder, '/', LOCK_FILE))

    md5_out_orig = as.character(md5(file(paste0(run_folder, '/', INFILL_OUT_FILE), open = "rb")))

    write('Sleeping...', file=paste0(run_folder, '/', log_r_file), append=TRUE)
    Sys.sleep(3)
    stuck_lock_count = 0
    stuck_md5_count = 0
    while (md5_out_orig == as.character(md5(file(paste0(run_folder, '/', INFILL_OUT_FILE), open = "rb")))) {
        if (stuck_md5_count >= MAX_WAIT_ROUND ) {
            break
        }
        
        if (file.exists(paste0(run_folder, '/', LOCK_FILE))) {
            if (stuck_lock_count >= MAX_WAIT_ROUND) {
                break
            }
            write(sprintf("File lock %s exists. Waiting for it to get processed...\n", LOCK_FILE), file=paste0(run_folder, '/', log_r_file), append=TRUE)
            Sys.sleep(2)
            stuck_lock_count = stuck_lock_count + 1
        } else {
            stuck_md5_count = stuck_md5_count + 1
        }
    }
    
    if (file.exists(paste0(run_folder, '/', LOCK_FILE)) && 
        md5_out_orig == 
        as.character(md5(file(paste0(run_folder, '/', INFILL_OUT_FILE), open = "rb")))
        ) 
    {
        # This should not happen
        stop("Inflling failed...")
    }

    # The lock doesn't exist anymore, so the response must be in the CSV file
    write('A new infill point will be generated soon.', file=paste0(run_folder, '/', log_r_file), append=TRUE)
    while (TRUE) {
        new_out_df = read.csv(paste0(run_folder, '/', INFILL_OUT_FILE), header=FALSE, skip=1)
        responses = new_out_df[, (NUM_PARAMS+1):ncol(new_out_df)]
        
        if (!is.na(responses[1])) {
                write(class(responses[1]), file=paste0(run_folder, '/', log_r_file), append=TRUE)
            
                text = sprintf("What am I? %s", as.character(responses[1]))
                write(text, file=paste0(run_folder, '/', log_r_file), append=TRUE)
                break            
        }
        Sys.sleep(2)
    }

    print(responses)
    write(as.character(responses), file=paste0(run_folder, '/', log_r_file), append=TRUE)
    write(class(responses), file=paste0(run_folder, '/', log_r_file), append=TRUE)

    responses = data.matrix(responses)
    return(responses)
}

write("NUM_CASES: ", file=paste0(run_folder, '/', log_r_file), append=TRUE)
write(class(NUM_CASES), file=paste0(run_folder, '/', log_r_file), append=TRUE)
write(NUM_CASES, file=paste0(run_folder, '/', log_r_file), append=TRUE)

if (NUM_CASES == 1) {
    # With only one response, we cannot use GPareto
    pso_optimized = pso::psoptim(rep(NA, NUM_PARAMS), fn = obj_fun, lower = rep(0, NUM_PARAMS), upper = rep(1, NUM_PARAMS), control=list(maxf=BUDGET))
    optimal_params = pso_optimized$par
    optimal_val = pso_optimized$value
    pso_optimized_dat = as.data.frame(t(as.matrix(c(optimal_params, optimal_val))))
    write.csv(pso_optimized_dat, paste0(run_folder, '/', 'pareto_front.csv'), quote=FALSE, row.names=FALSE)
} else {
    design_init_file = config$infill$r_design_init_out_file
    write("(R subsystem) Reading in design init csv file and run OBJ FUN", file=paste0(run_folder, '/', log_r_file), append=TRUE)
    design_init = read.csv(paste0(run_folder, '/', design_init_file))
    design_init = design_init[,-1]

    write(dim(design_init), file=paste0(run_folder, '/', log_r_file), append=TRUE)
    design_init_response_file = config$infill$r_design_init_response_out_file
    write("(R subsystem) Reading in design init responses csv file and run OBJ FUN", file=paste0(run_folder, '/', log_r_file), append=TRUE)
    design_init_reponses = read.csv(paste0(run_folder, '/', design_init_response_file), header = FALSE, skip = 1)
    write(dim(design_init_reponses), file=paste0(run_folder, '/', log_r_file), append=TRUE)
    write(as.character(design_init_reponses), file=paste0(run_folder, '/', log_r_file), append=TRUE)

    if (nrow(design_init) != nrow(design_init_reponses)) {
        stop(sprintf("The design has %d rows, but there are %d responses. Try to rerun the design matrix", nrow(design_init), nrow(design_init_reponses)))
    }
    
    # Where the action is...
    gp_pso = easyGParetoptim(fn=obj_fun, par=design_init,
        value=design_init_reponses, lower=rep(0, NUM_PARAMS), 
        upper= rep(1, NUM_PARAMS), budget=BUDGET, 
        control=list(method=METHOD, trace=2, 
        inneroptim=ALG, maxit=MAXIT, refPoint=R_REF_POINT)
    )

    pfronts_pso = gp_pso$par
    write.csv(pfronts_pso, paste0(run_folder, '/', 'pareto_front.csv'), quote=FALSE, row.names=FALSE)

    # Save the surrogate model
    save(gp_pso, file=paste0(run_folder, '/', 'surrogate_model.RData'))
}
write("(R subsystem) Infilling is complete.\n", file=paste0(run_folder, '/', log_r_file), append=TRUE)
