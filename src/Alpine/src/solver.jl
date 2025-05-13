# Optimizer struct and options to tune Alpine, a global solver for MINLPs

mutable struct OptimizerOptions
    # Parameters for tuning Alpine

    # Basic solver parameters
    log_level   :: Int                                             # Verbosity flag: 0 for quiet, 1 for basic solve info, 2 for iteration info
    time_limit  :: Float64                                         # Time limit for algorithm (in seconds)
    max_iter    :: Int                                             # Maximum bound on number of iterations for partitioning algorithm
    rel_gap     :: Float64                                         # Relative optimality gap for termination condition
    abs_gap     :: Float64                                         # Absolute optimality gap for termination condition
    tol         :: Float64                                         # Numerical tol used in algorithms
    large_bound :: Float64                                         # Large bounds for problems with unbounded variables

    # All the solver options
    nlp_solver   :: Union{MOI.OptimizerWithAttributes, Nothing}    # Local continuous NLP solver for solving NLPs at each iteration
    minlp_solver :: Union{MOI.OptimizerWithAttributes, Nothing}    # Local MINLP solver for solving MINLPs at each iteration
    mip_solver   :: Union{MOI.OptimizerWithAttributes, Nothing}    # MIP solver for successive lower bound solves

    # Convexification methods
    recognize_convex    :: Bool                                    # Recognize convex expressions in parsing objective functions and constraints
    bilinear_mccormick  :: Bool                                    # Convexify bilinear terms using piecwise McCormick representation
    bilinear_convexhull :: Bool                                    # Convexify bilinear terms using lambda representation
    monomial_convexhull :: Bool                                    # Convexify monomial terms using convex-hull representation

    # Expression-based user-inputs
    # method_convexification :: Array{Function}                    # Array of functions that user can choose to convexify specific non-linear terms : no over-ride privilege

    # Parameters used in Alpine's MIP-based partitioning algorithm
    apply_partitioning            :: Bool                          # Apply the partitioning algorithm only if thhis true, else terminate after presolve
    disc_var_pick                 :: Any                           # Algorithm for choosing the variables to discretize: 1 for minimum vertex cover, 0 for all variables
    disc_ratio                    :: Any                           # Discretization ratio parameter, which is critical for convergence (using a fixed value for now, later switch to a function)
    disc_uniform_rate             :: Int                           # Discretization rate parameter when using uniform partitions
    disc_add_partition_method     :: Any                           # Additional methods to add discretization
    disc_divert_chunks            :: Int                           # How many uniform partitions to construct
    disc_abs_width_tol            :: Float64                       # Absolute tolerance used when setting up a partition
    disc_rel_width_tol            :: Float64                       # Relative width tolerance when setting up a partition
    disc_consecutive_forbid       :: Int                           # Prevent bounding model to add partitions consecutively in the same region when bounds do not improve
    disc_ratio_branch             :: Bool                          # Branching tests for picking fixed discretization ratio
    
    # ADDED PARAMETERS
    disc_ratio_iter1              :: Any                           # Use a different disc_ratio for the first iteration, if needed
    cutoff_bounding_solve         :: Bool                          # Use a Cutoff while solving the MIP bounding problem?
    exclude_con_local_solve       :: Dict                          # Indices of linear constraints to exclude from the local solve
    outer_approximate_quadratics  :: Bool                          # Should we outer-approximate the convex relaxation of quadratic terms?
    disc_use_single_partitions    :: Bool                          # Use single partitions at the bounding solution at each iteration?
    disc_partition_at_existing    :: Bool                          # Do we add partitions for a variable whose bounding solution is at a discretization point?
	disc_sp_iter                  :: Int                           # For how many iterations of Alpine should we solve max-min problem to add partitions? 
    disc_sp_num_points            :: Int                           # Number of partition points per variable to be determined by solving the max-min problem
    disc_sp_adaptive_part         :: Bool                          # Heuristic to choose number of partition points per variable adaptively based on the solution of the bounding problem
    disc_sp_min_spacing_factor    :: Float64                       # Minimum spacing between discretization points (including bounds) relative to the variable width
    disc_sp_method                :: String                        # Method for solving the max-min problem. Options: "mpbngc" and "nlopt"
    disc_sp_budget                :: Int                           # Limit for number of max-min solves at every bounding iteration of global solve 
    disc_sp_postprocessing        :: Bool                          # Check if any of the SP discretization points are not useful and make trivial. Could be expensive
    disc_sp_postprocessing_pert   :: Bool                          # Do a perturbation analysis of the postprocessed MPBNGC solution. Could be expensive
    disc_sp_time_limit            :: Float64                       # Time Limit for max-min solves at every bounding iteration of global solve 
    disc_use_initial_guess        :: Bool                          # Use the user-specified partitioning fractions as an initial guess for the max-min problem?
    disc_specified_initial_guess  :: Dict                          # User-specified initial guess for the discretization fractions for each SP iteration
    disc_use_specified_part       :: Bool                          # Use the user-specified partitioning fractions instead of solving the max-min problem?
    disc_specified_part           :: Dict                          # User-specified partition fractions for each iteration until disc_sp_iter

    # MIP formulation parameters
    convhull_formulation          :: String                        # MIP Formulation for the relaxation
    convhull_ebd                  :: Bool                          # Enable embedding formulation
    convhull_ebd_encode           :: Any                           # Encoding method used for convhull_ebd
    convhull_ebd_ibs              :: Bool                          # Enable independent branching scheme
    convhull_ebd_link             :: Bool                          # Linking constraints between x and α, type 1 uses hierarchical and type 2 uses big-m
    convhull_warmstart            :: Bool                          # Warm start the bounding MIP
    convhull_no_good_cuts         :: Bool                          # Add no-good cuts to MIP based on the pool solutions

    # Bound-tightening parameters
    presolve_track_time           :: Bool                          # Account presolve time for total time usage
    presolve_bt                   :: Bool                          # Perform bound tightening procedure before the main algorithm (default: true)
    presolve_bt_time_limit        :: Float64                       # Time limit for presolving (seconds)
    presolve_bt_max_iter          :: Int                           # Maximum iterations allowed to perform presolve
    presolve_bt_width_tol         :: Float64                       # Width tolerance for bound-tightening
    presolve_bt_improv_tol        :: Float64                       # Improvement tolerance for average reduction of variable ranges
    presolve_bt_bound_tol         :: Float64                       # Variable bounds truncation precicision tol
    presolve_bt_obj_bound_tol     :: Float64                       # Objective upper bound truncation tol
    presolve_bt_algo              :: Any                           # Method used for bound tightening procedures, can either be an index of default methods or functional inputs
    presolve_bt_relax_integrality :: Bool                          # Relax the MIP solved in built-in relaxation scheme for time performance
    presolve_bt_mip_time_limit    :: Float64                       # Time limit for a single MIP solved in the built-in bound tightening algorithm (with partitions)

    # Domain Reduction
    presolve_bp                   :: Bool                          # Conduct basic bound propagation

    # Features for integer variable problems
    int_enable                    :: Bool                          # Convert integer problem into binary problem
    int_cumulative_disc           :: Bool                          # Cumulatively involve integer variables for discretization

end

function get_default_options()
        log_level   = 1
        time_limit  = 1E6
        max_iter    = 99
        rel_gap     = 1e-4
        abs_gap     = 1e-6
        tol         = 1e-6
        large_bound = 1E6

        nlp_solver   = nothing
        minlp_solver = nothing
        mip_solver   = nothing

        recognize_convex    = true
        bilinear_mccormick  = false
        bilinear_convexhull = true
        monomial_convexhull = true

        # method_convexification = Array{Function}(undef, 0)
        # method_partition_injection = Array{Function}(undef, 0)
        # term_patterns = Array{Function}(undef, 0)
        # constr_patterns = Array{Function}(undef, 0)

        apply_partitioning            = true
        disc_var_pick                 = 2                # By default, uses the 15-variable selective rule
        disc_ratio                    = 10
        disc_uniform_rate             = 2
        disc_add_partition_method     = "adaptive"
        disc_divert_chunks            = 5
        disc_abs_width_tol            = 1e-4
        disc_rel_width_tol            = 1e-6
        disc_consecutive_forbid       = false
        disc_ratio_branch             = false

        # ADDED PARAMETERS INITIALIZATION
        disc_ratio_iter1              = -1
        cutoff_bounding_solve         = false
        exclude_con_local_solve       = Dict()
        outer_approximate_quadratics  = false
        disc_use_single_partitions    = false
        disc_partition_at_existing    = true
        disc_sp_iter                  = -1               # By default, do not use use SP
        disc_sp_num_points            = 2
        disc_sp_adaptive_part         = false
        disc_sp_min_spacing_factor    = 100.0
        disc_sp_method                = "mpbngc"
        disc_sp_budget                = 200
        disc_sp_postprocessing        = true
        disc_sp_postprocessing_pert   = false
        disc_sp_time_limit            = time_limit
        disc_use_initial_guess        = false
        disc_specified_initial_guess  = Dict()
        disc_use_specified_part       = false
        disc_specified_part           = Dict()
    
        convhull_formulation          = "sos2"
        convhull_ebd                  = false
        convhull_ebd_encode           = "default"
        convhull_ebd_ibs              = false
        convhull_ebd_link             = false
        convhull_warmstart            = true
        # ADDED
        convhull_no_good_cuts         = false

        presolve_track_time           = true
        presolve_bt                   = true
        presolve_bt_time_limit        = 900
        presolve_bt_max_iter          = 25
        presolve_bt_width_tol         = 1e-2
        presolve_bt_improv_tol        = 1e-3
        presolve_bt_bound_tol         = 1e-4
        presolve_bt_obj_bound_tol     = 1e-2
        presolve_bt_algo              = 1
        presolve_bt_relax_integrality = false
        presolve_bt_mip_time_limit    = Inf
        presolve_bp                   = false

        int_enable                    = false
        int_cumulative_disc           = true

    return OptimizerOptions(log_level, 
                            time_limit, 
                            max_iter, 
                            rel_gap, 
                            abs_gap, 
                            tol, 
                            large_bound,
                            nlp_solver, 
                            minlp_solver, 
                            mip_solver,
                            recognize_convex, 
                            bilinear_mccormick, 
                            bilinear_convexhull, 
                            monomial_convexhull,
                            apply_partitioning,
                            disc_var_pick, 
                            disc_ratio, 
                            disc_uniform_rate, 
                            disc_add_partition_method,
                            disc_divert_chunks,
                            disc_abs_width_tol, 
                            disc_rel_width_tol, 
                            disc_consecutive_forbid, 
                            disc_ratio_branch,
                            disc_ratio_iter1,  # ADDED START
                            cutoff_bounding_solve,
                            exclude_con_local_solve,
                            outer_approximate_quadratics,
                            disc_use_single_partitions,
                            disc_partition_at_existing,
                            disc_sp_iter,
                            disc_sp_num_points,
                            disc_sp_adaptive_part,
                            disc_sp_min_spacing_factor,
                            disc_sp_method,
                            disc_sp_budget,
                            disc_sp_postprocessing,
                            disc_sp_postprocessing_pert,
                            disc_sp_time_limit,
                            disc_use_initial_guess,
                            disc_specified_initial_guess,
                            disc_use_specified_part,
                            disc_specified_part,  # ADDED END
                            convhull_formulation, 
                            convhull_ebd, 
                            convhull_ebd_encode, 
                            convhull_ebd_ibs, 
                            convhull_ebd_link, 
                            convhull_warmstart, 
                            convhull_no_good_cuts,
                            presolve_track_time, 
                            presolve_bt, 
                            presolve_bt_time_limit, 
                            presolve_bt_max_iter, 
                            presolve_bt_width_tol, 
                            presolve_bt_improv_tol, 
                            presolve_bt_bound_tol,
                            presolve_bt_obj_bound_tol, 
                            presolve_bt_algo, 
                            presolve_bt_relax_integrality, 
                            presolve_bt_mip_time_limit, 
                            presolve_bp, 
                            int_enable, 
                            int_cumulative_disc)
                            
end
