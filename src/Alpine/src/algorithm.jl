const STATUS_LIMIT = [
    MOI.ITERATION_LIMIT, MOI.TIME_LIMIT, MOI.NODE_LIMIT,
    MOI.SOLUTION_LIMIT, MOI.MEMORY_LIMIT, MOI.OBJECTIVE_LIMIT,
    MOI.NORM_LIMIT, MOI.OTHER_LIMIT
]
const STATUS_OPT = [
    MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED
]
const STATUS_INF = [
    MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE
]

function features_available(m::Optimizer)
   features = [:Grad, :Jac, :JacVec, :ExprGraph]
   if !m.disable_hessian
       push!(features, :Hess)
       push!(features, :HessVec)
   end

   return features
end

function load!(m::Optimizer)
   
   # Initialize NLP interface 
   requested_features = Alp.features_available(m)
   if m.d_orig !== nothing 
      MOI.initialize(m.d_orig, requested_features::Vector{Symbol})
   end 

   for feat in requested_features
      if !(feat in Alp.features_available(m))
          error("Unsupported feature $feat")
      end
  end

   # Collect objective & constraint expressions
   if m.has_nl_objective
       m.obj_expr_orig = Alp.expr_isolate_const(_variable_index_to_index(MOI.objective_expr(m.d_orig))) # see in nlexpr.jl if this expr isolation has any issue
   elseif m.objective_function isa Nothing
       m.obj_expr_orig = Expr(:call, :+)
   else
       m.obj_expr_orig = _moi_function_to_expr(m.objective_function)
   end

   # Collect original variable type and build dynamic variable type space
   m.var_type = copy(m.var_type_orig)
   m.int_vars = [i for i in 1:m.num_var_orig if m.var_type[i] == :Int]
   m.bin_vars = [i for i in 1:m.num_var_orig if m.var_type[i] == :Bin]

   if !isempty(m.int_vars) || !isempty(m.bin_vars)
       (Alp.get_option(m, :minlp_solver) === nothing) && (error("No MINLP local solver specified; use option 'minlp_solver' to specify a MINLP local solver"))
   end

   m.num_constr_orig += length(m.nl_constraint_bounds_orig)
   m.num_nlconstr_orig += length(m.nl_constraint_bounds_orig)
   append!(m.constraint_bounds_orig, m.nl_constraint_bounds_orig)
   for i in eachindex(m.nl_constraint_bounds_orig)
       push!(m.constr_expr_orig, _variable_index_to_index(MOI.constraint_expr(m.d_orig, i)))
       push!(m.constr_structure, :generic_nonlinear)
   end

   # Summarize constraints information in original model
   m.constr_type_orig = Array{Symbol}(undef, m.num_constr_orig)

   for i in 1:m.num_constr_orig
       if m.constraint_bounds_orig[i].lower > -Inf && m.constraint_bounds_orig[i].upper < Inf
           m.constr_type_orig[i] = :(==)
       elseif m.constraint_bounds_orig[i].lower > -Inf
           m.constr_type_orig[i] = :(>=)
       else
           m.constr_type_orig[i] = :(<=)
       end
   end

   # Initialize recognizable structure properties with :none
   m.obj_structure = :none

   @assert m.num_constr_orig == m.num_nlconstr_orig + m.num_lconstr_orig
   m.is_obj_linear_orig = !m.has_nl_objective && m.objective_function isa MOI.ScalarAffineFunction{Float64}
   m.is_obj_linear_orig ? (m.obj_structure = :generic_linear) : (m.obj_structure = :generic_nonlinear)
   isa(m.obj_expr_orig, Number) && (m.obj_structure = :constant)

   # populate data to create the bounding model
   Alp.recategorize_var(m)             # Initial round of variable re-categorization

   :Int in m.var_type_orig && error("Alpine does not support MINLPs with generic integer (non-binary) variables yet! Try Juniper.jl for finding a local feasible solution")
   :Int in m.var_type_orig ? Alp.set_option(m, :int_enable, true) : Alp.set_option(m, :int_enable, false) # Separator for safer runs

   # Conduct solver-dependent detection
   # ADDED
   # Alp._fetch_mip_solver_identifier(m)
   # (Alp.get_option(m, :nlp_solver) !== nothing)   && (Alp._fetch_nlp_solver_identifier(m))
   # (Alp.get_option(m, :minlp_solver) !== nothing) && (Alp._fetch_minlp_solver_identifier(m))

   # ADDED
   Alp._fetch_mip_solver_identifier(m)
   Alp._fetch_nlp_solver_identifier(m)
   Alp._fetch_minlp_solver_identifier(m)

   # Solver Dependent Options
   if m.mip_solver_id != :Gurobi
       Alp.get_option(m, :convhull_warmstart) == false
       Alp.get_option(m, :convhull_no_good_cuts) == false
   end

   # Main Algorithmic Initialization
   Alp.process_expr(m)                         # Compact process of every expression
   Alp.init_tight_bound(m)                     # Initialize bounds for algorithmic processes
   Alp.resolve_var_bounds(m)                   # resolve lifted var bounds
   Alp.pick_disc_vars(m)                       # Picking variables to be discretized
   Alp.init_disc(m)                            # Initialize discretization dictionaries

   # Turn-on bt presolve if variables are not discrete
   if isempty(m.int_vars) && length(m.bin_vars) <= 50 && m.num_var_orig <= 10000 && length(m.candidate_disc_vars)<=300 && Alp.get_option(m, :presolve_bt) == nothing
       Alp.set_option(m, :presolve_bt, true)
       println("Automatically turning on bound-tightening presolve")
   elseif Alp.get_option(m, :presolve_bt) == nothing  # If no use indication
       Alp.set_option(m, :presolve_bt, false)
   end

   if length(m.bin_vars) > 200 || m.num_var_orig > 2000
       println("Automatically turning OFF 'disc_ratio_branch' due to the size of the problem")
       Alp.set_option(m, :disc_ratio_branch, false)
   end

   # Initialize the solution pool
   m.bound_sol_pool = Alp.initialize_solution_pool(m, 0)  # Initialize the solution pool

   # Check if any illegal term exist in the warm-solution
   any(isnan, m.best_sol) && (m.best_sol = zeros(length(m.best_sol)))

   # Initialize log
   Alp.logging_summary(m)

   return
end


"""
   High-level Function
"""
function MOI.optimize!(m::Optimizer)
   Alp.load!(m)
   if getproperty(m, :presolve_infeasible)
      Alp.summary_status(m)
      return
   end

   Alp.presolve(m)

   if !Alp.check_exit(m) && Alp.get_option(m, :apply_partitioning)
      Alp.global_solve(m)
      Alp.get_option(m, :log_level)  > 0 && Alp.logging_row_entry(m, finish_entry=true)
      println("====================================================================================================")
   else
      # ADDED
      if m.status[:bounding_solve] == MOI.NUMERICAL_ERROR
         if m.detected_feasible_solution && m.detected_bound
            println("  Presolve terminated with a feasible solution and a bound")
         elseif m.detected_feasible_solution
            println("  Presolve terminated only with a feasible solution (no bound)")
         elseif m.detected_bound
            println("  Presolve terminated only with a bound (no feasible solution)")
         else
            println("  Presolve terminated without a feasible solution and a bound")
         end
      else
         println("  Presolve terminated with a global optimal solution")
      end
   end
   
   Alp.summary_status(m)
   
   return
end

"""
   global_solve(m::Optimizer)

Perform global optimization algorithm that is based on the adaptive piecewise convexification.
This iterative algorithm loops over [`bounding_solve`](@ref) and [`local_solve`](@ref) until the optimality gap between the lower bound (relaxed problem with min. objective) and the upper bound (feasible problem) is within the user prescribed limits.
Each [`bounding_solve`](@ref) provides a lower bound that serves as the partitioning point for the next iteration (this feature can be modified given a different `add_adaptive_partition`).
Each [`local_solve`](@ref) provides an incumbent feasible solution. The algorithm terminates when atleast one of these conditions are satisfied: time limit, optimality condition, or iteration limit.

"""
function global_solve(m::Optimizer)

   Alp.get_option(m, :log_level) > 0 && Alp.logging_head(m)
   Alp.get_option(m, :presolve_track_time) || Alp.reset_timer(m)
   while !Alp.check_exit(m)
      m.logs[:n_iter] += 1
      Alp.create_bounding_mip(m)                     # Build the relaxation model
      Alp.bounding_solve(m)                          # Solve the relaxation model
      Alp.update_opt_gap(m)                          # Update optimality gap
      Alp.check_exit(m) && break                     # Feasibility check
      Alp.get_option(m, :log_level) > 0 && Alp.logging_row_entry(m)  # Logging
      Alp.local_solve(m)                             # Solve local model for feasible solution
      Alp.local_solve(m, init_at_bound_sol = true)   # ADDED: Solve local model with bounding solution as initial guess
      Alp.update_opt_gap(m)                          # Update optimality gap
      Alp.check_exit(m) && break                     # Detect optimality termination
      Alp.algorithm_automation(m)                    # Automated adjustments
      Alp.add_partition(m)                           # Add extra discretizations
   end

   return
end

"""
   presolve(m::Optimizer)
"""
function presolve(m::Optimizer)
   start_presolve = time()
   Alp.get_option(m, :log_level) > 0 && printstyled("PRESOLVE \n", color=:cyan)
   Alp.get_option(m, :log_level) > 0 && println("  Doing local search")
   Alp.local_solve(m, presolve = true)

   # ADDED
   # println("local solver status: ", m.status[:local_solve])

   # Solver status
   if m.status[:local_solve] in STATUS_OPT || m.status[:local_solve] in STATUS_LIMIT

      Alp.get_option(m, :log_level) > 0 && println("  Local solver returns a feasible point with value $(round(m.best_obj, digits=4))")
      Alp.bound_tightening(m, use_bound = true)    # performs bound-tightening with the local solve objective value
      Alp.get_option(m, :presolve_bt) && init_disc(m)            # Re-initialize discretization dictionary on tight bounds
      Alp.get_option(m, :disc_ratio_branch) && (Alp.set_option(m, :disc_ratio, Alp.update_disc_ratio(m, true)))
      Alp.add_partition(m, use_solution=m.best_sol)  # Setting up the initial discretization

   elseif m.status[:local_solve] in STATUS_INF

      (Alp.get_option(m, :log_level) > 0) && println("  Bound tightening without objective bounds (OBBT)")
      Alp.bound_tightening(m, use_bound = false)                      # do bound tightening without objective value
      (Alp.get_option(m, :disc_ratio_branch)) && (Alp.set_option(m, :disc_ratio, Alp.update_disc_ratio(m, true)))
      Alp.get_option(m, :presolve_bt) && Alp.init_disc(m)

      # ADDED
      println("Local solve did not yield a feasible solution during presolve")
      if Alp.get_option(m, :disc_sp_iter) > 0
         Alp.add_partition(m)
      end

   elseif m.status[:local_solve] == MOI.INVALID_MODEL

      @warn " Warning: Presolve ends with local solver yielding $(m.status[:local_solve]). \n This may come from Ipopt's `:Not_Enough_Degrees_Of_Freedom`. \n Consider more replace equality constraints with >= and <= to resolve this."

   else

      @warn " Warning: Presolve ends with local solver yielding $(m.status[:local_solve])."

      # ADDED
      (Alp.get_option(m, :disc_ratio_branch)) && (Alp.set_option(m, :disc_ratio, Alp.update_disc_ratio(m, true)))
      Alp.get_option(m, :presolve_bt) && Alp.init_disc(m)
      println("Local solve did not yield a feasible solution during presolve")
      if Alp.get_option(m, :disc_sp_iter) > 0
         Alp.add_partition(m)
      end

   end

   cputime_presolve = time() - start_presolve
   m.logs[:presolve_time] += (cputime_presolve - m.logs[:max_min_time])
   m.logs[:total_time] = m.logs[:presolve_time]
   m.logs[:time_left] -= m.logs[:presolve_time]
   
   if Alp.get_option(m, :presolve_bt)
      (Alp.get_option(m, :log_level) > 0) && println("  Post-presolve optimality gap: $(round(m.presolve_best_rel_gap; digits = 3))%")
   end
   (Alp.get_option(m, :log_level) > 0) && println("  Completed presolve in $(round.(m.logs[:total_time]; digits = 2))s")
   
   return
end

"""
   A wrapper function that collects some automated solver adjustments within the main while loop.
"""
function algorithm_automation(m::Optimizer)

   Alp.get_option(m, :disc_var_pick) == 3 && Alp.update_disc_cont_var(m)
   Alp.get_option(m, :int_cumulative_disc) && Alp.update_disc_int_var(m)

   if Alp.get_option(m, :disc_ratio_branch)
      Alp.set_option(m, :disc_ratio, Alp.update_disc_ratio(m, true))    # Only perform for a maximum three times
   end

   return
end

"""
   Summarized function to determine whether to interrupt the main while loop.
"""
function check_exit(m::Optimizer)

   # constant objective with feasible local solve check
   if Alp.expr_isconst(m.obj_expr_orig) && (m.status[:local_solve] == MOI.OPTIMAL || m.status == MOI.LOCALLY_SOLVED)
      # m.best_bound = eval(m.obj_expr_orig)
      m.best_bound = m.obj_expr_orig
      m.best_rel_gap = 0.0
      m.best_abs_gap = 0.0
      m.status[:bounding_solve] = MOI.OPTIMAL
      m.alpine_status = :Optimal
      m.detected_bound = true
      return true
   end

   # Infeasibility check
   m.status[:bounding_solve] == MOI.INFEASIBLE      && return true

   # Unbounded check
   m.status[:bounding_solve] == MOI.DUAL_INFEASIBLE && return true

   # ADDED
   # Check for numerical issues during partitioning
   m.status[:bounding_solve] == MOI.NUMERICAL_ERROR && return true

   # Optimality check
   if m.best_rel_gap <= Alp.get_option(m, :rel_gap)
      m.detected_bound = true
      return true
   end
   m.logs[:n_iter] >= Alp.get_option(m, :max_iter)  && return true
   m.best_abs_gap <= Alp.get_option(m, :abs_gap)    && return true

   # User-limits check
   m.logs[:time_left] < Alp.get_option(m, :tol)     && return true

   return false
end

function load_nonlinear_model(m::Optimizer, model::MOI.ModelLike, l_var, u_var)
    x = MOI.add_variables(model, m.num_var_orig)
    for i in eachindex(x)
        set = Alp._bound_set(l_var[i], u_var[i])
        if set !== nothing
            MOI.add_constraint(model, x[i], set)
        end
    end
    # ADDED
    count = 0
    for (func, set) in m.lin_quad_constraints
      # ADDED
      count += 1
      if haskey(Alp.get_option(m, :exclude_con_local_solve), string(count))
         # do nothing
      else
         MOI.add_constraint(model, func, set)
      end
    end
    MOI.set(model, MOI.ObjectiveSense(), m.sense_orig)
    if m.objective_function !== nothing
        MOI.set(model, MOI.ObjectiveFunction{typeof(m.objective_function)}(), m.objective_function)
    end
    if m.d_orig !== nothing 
       block = MOI.NLPBlockData(m.nl_constraint_bounds_orig, m.d_orig, m.has_nl_objective)
       MOI.set(model, MOI.NLPBlock(), block)
    end

    return x
end

function set_variable_type(model::MOI.ModelLike, xs, variable_types)
    for (x, variable_type) in zip(xs, variable_types)
        if variable_type == :Int
            MOI.add_constraint(model, x, MOI.Integer())
        elseif variable_type == :Bin
            MOI.add_constraint(model, x, MOI.ZeroOne())
        else
            @assert variable_type == :Cont
        end
    end
end

"""
   Alp.local_solve(m::Optimizer, presolve::Bool=false)

Perform a local NLP or MINLP solve to obtain a feasible solution.
The `presolve` option is set to `true` when the function is invoked in [`presolve`](@ref).
Otherwise, the function is invoked from [`bounding_solve`](@ref).

"""
# function local_solve(m::Optimizer; presolve = false)
# ADDED
function local_solve(m::Optimizer; presolve = false, init_at_bound_sol = false)

   # ADDED
   # println("\n Starting local solve")

   # ADDED
   if m.nlp_solver_id == "Knitro"
      KN_LMC = KNITRO.LMcontext()
      knitro_solver = JuMP.optimizer_with_attributes(
                        () -> KNITRO.Optimizer(license_manager=KN_LMC),
                        "algorithm" => 3,
                        MOI.Silent() => true
                     )
   end

   convertor = Dict(MOI.MAX_SENSE => :>, MOI.MIN_SENSE => :<)
   local_nlp_status = :Unknown

   var_type_screener = [i for i in m.var_type_orig if i in [:Bin, :Int]]

   # ADDED
   #=
   if presolve
      if !isempty(var_type_screener) && Alp.get_option(m, :minlp_solver) !== nothing
         local_solve_model = MOI.instantiate(Alp.get_option(m, :minlp_solver), with_bridge_type=Float64)
      elseif !isempty(var_type_screener)
         local_solve_model = MOI.instantiate(Alp.get_option(m, :nlp_solver), with_bridge_type=Float64)
      else
         local_solve_model = MOI.instantiate(Alp.get_option(m, :nlp_solver), with_bridge_type=Float64)
      end
   else
      local_solve_model = MOI.instantiate(Alp.get_option(m, :nlp_solver), with_bridge_type=Float64)
   end
   =#

   if presolve
      if !isempty(var_type_screener) && Alp.get_option(m, :minlp_solver) !== nothing
         local_solve_model = MOI.instantiate(Alp.get_option(m, :minlp_solver), with_bridge_type=Float64)
      elseif !isempty(var_type_screener)
         if m.nlp_solver_id == "Knitro"
            local_solve_model = MOI.instantiate(knitro_solver, with_bridge_type=Float64)
         else
            local_solve_model = MOI.instantiate(Alp.get_option(m, :nlp_solver), with_bridge_type=Float64)
         end
      else
         if m.nlp_solver_id == "Knitro"
            local_solve_model = MOI.instantiate(knitro_solver, with_bridge_type=Float64)
         else
            local_solve_model = MOI.instantiate(Alp.get_option(m, :nlp_solver), with_bridge_type=Float64)
         end
      end
   else
      if m.nlp_solver_id == "Knitro"
         local_solve_model = MOI.instantiate(knitro_solver, with_bridge_type=Float64)
      else
         local_solve_model = MOI.instantiate(Alp.get_option(m, :nlp_solver), with_bridge_type=Float64)
      end
   end

   # if presolve == false
   # ADDED
   if (!presolve) && (!init_at_bound_sol)
      l_var, u_var = Alp.fix_domains(m)
   else
      l_var, u_var = m.l_var_tight[1:m.num_var_orig], m.u_var_tight[1:m.num_var_orig]
   end

   x = Alp.load_nonlinear_model(m, local_solve_model, l_var, u_var)

   if m.d_orig !== nothing 
      (!m.d_orig.want_hess) && MOI.initialize(m.d_orig, [:Grad, :Jac, :Hess, :HessVec, :ExprGraph]) # Safety scheme for sub-solvers re-initializing the NLPEvaluator
   end
#=
   if !presolve
      warmval = m.best_sol[1:m.num_var_orig]
   else
      warmval = m.initial_warmval[1:m.num_var_orig]
   end
=#
   # ADDED
   if (!presolve) && (!init_at_bound_sol)
      warmval = m.best_sol[1:m.num_var_orig]
   elseif presolve
      warmval = m.initial_warmval[1:m.num_var_orig]
   else
      warmval = m.best_bound_sol[1:m.num_var_orig]
   end
   MOI.set(local_solve_model, MOI.VariablePrimalStart(), x, warmval)

   # do_heuristic = false

   # The only case when MINLP solver is actually used
   if presolve && !isempty(var_type_screener)
      if Alp.get_option(m, :minlp_solver) === nothing
         error("Provide a valid MINLP solver")
         # do_heuristic = true
      else
         Alp.set_variable_type(local_solve_model, x, m.var_type_orig)
      end
   end

   # ADDED
   # if m.logs[:n_iter] == 0
   #    println("local solve model: ", local_solve_model)
   # end

   start_local_solve = time()
   MOI.optimize!(local_solve_model)
   local_nlp_status = MOI.get(local_solve_model, MOI.TerminationStatus())

   # ADDED
   # println("local solver status: ", local_nlp_status)
   # println("local solver primal status: ", MOI.get(local_solve_model,MOI.PrimalStatus()))

   # if !do_heuristic
   #     local_nlp_status = MOI.get(local_solve_model, MOI.TerminationStatus())
   # end


   cputime_local_solve = time() - start_local_solve
   m.logs[:total_time] += cputime_local_solve
   m.logs[:local_solves_time] += cputime_local_solve
   m.logs[:time_left] = max(0.0, Alp.get_option(m, :time_limit) - m.logs[:total_time])

   # if do_heuristic
      # m.status[:local_solve] = heu_basic_rounding(m, MOI.get(local_solve_model, MOI.VariablePrimal(), x))
      # return

   if local_nlp_status in STATUS_OPT || local_nlp_status in STATUS_LIMIT
      candidate_obj = MOI.get(local_solve_model, MOI.ObjectiveValue())
      sol_temp = MOI.get(local_solve_model, MOI.VariablePrimal(), x)
      candidate_sol = Vector{Float64}()

      # ADDED
      # println("  Local solver returns a feasible point with value $(round(candidate_obj, digits=4))")

      feas_tol = 1E-5

      for i = 1:length(sol_temp)
         if (sol_temp[i] >= m.l_var_orig[i] - feas_tol) && (sol_temp[i] <= m.l_var_orig[i] + feas_tol)
            push!(candidate_sol, m.l_var_orig[i])
         elseif (sol_temp[i] >= m.u_var_orig[i] - feas_tol) && (sol_temp[i] <= m.u_var_orig[i] + feas_tol)
            push!(candidate_sol, m.u_var_orig[i])
         else
            # push!(candidate_sol, round(sol_temp[i], digits = 7))
            # ADDED
            push!(candidate_sol, sol_temp[i])
         end
      end

      @assert length(candidate_sol) == length(sol_temp)
      Alp.update_incumb_objective(m, candidate_obj, candidate_sol)
      m.status[:local_solve] = local_nlp_status

      # ADDED
      # println("Finished local solve \n")      
      if m.logs[:n_iter] == 0
         println("\nFeasible solution found during presolve: ",candidate_sol,"  :with objective: ",candidate_obj,"\n")
      end

      # ADDED
      if m.nlp_solver_id == "Knitro"
         KNITRO.KN_release_license(KN_LMC)
      end

      return

   elseif local_nlp_status in STATUS_INF
      Alp.heu_pool_multistart(m) == MOI.LOCALLY_SOLVED && return
      push!(m.logs[:obj], "INF")
      m.status[:local_solve] = MOI.LOCALLY_INFEASIBLE
      
      # ADDED
      # println("Finished local solve \n")

      # ADDED
      if m.nlp_solver_id == "Knitro"
         KNITRO.KN_release_license(KN_LMC)
      end

      return

   elseif local_nlp_status == MOI.DUAL_INFEASIBLE
      push!(m.logs[:obj], "U")
      m.status[:local_solve] = MOI.DUAL_INFEASIBLE
      if presolve
         @warn "  Warning: NLP local solve is unbounded."
      else
         @warn "  Warning: NLP local solve is unbounded."
      end
      
      # ADDED
      # println("Finished local solve \n")

      # ADDED
      if m.nlp_solver_id == "Knitro"
         KNITRO.KN_release_license(KN_LMC)
      end

      return

   else
      push!(m.logs[:obj], "E")
      m.status[:local_solve] = MOI.OTHER_ERROR
      if presolve
         @warn " Warning: NLP solve failure $(local_nlp_status)."
      else
         @warn " Warning: NLP local solve failure."
      end
      
      # ADDED
      # println("Finished local solve \n")

      # ADDED
      if m.nlp_solver_id == "Knitro"
         KNITRO.KN_release_license(KN_LMC)
      end

      return
   end

   return
end


"""
   Alp.bounding_solve(m::Optimizer; kwargs...)

This step usually solves a convex MILP/MIQCP/MIQCQP problem for lower bounding the given minimization problem.
It solves the problem built upon a piecewise convexification based on the discretization sictionary of some variables.
See `create_bounding_mip` for more details of the problem solved here.
"""
# function bounding_solve(m::Optimizer)
# ADDED
function bounding_solve(m::Optimizer; kwargs...)
      
   # ADDED
   # println("\n Starting bounding solve")

   # ADDED
   options = Dict(kwargs)
   haskey(options, :add_time) ? add_time = options[:add_time] : add_time = true

   convertor = Dict(MOI.MAX_SENSE => :<, MOI.MIN_SENSE => :>)

   # Updates time metric and the termination bounds
   Alp.set_mip_time_limit(m)
   # update_boundstop_options(m)

   # ADDED
   if Alp.get_option(m, :cutoff_bounding_solve)
      tol = Alp.get_option(m, :tol)
      if !isapprox(m.best_obj, 0.0; atol = tol)
         if Alp.is_min_sense(m)
            # cutoff_value = m.best_obj - abs(m.best_obj)*Alp.get_option(m, :rel_gap)
            cutoff_value = m.best_obj + abs(m.best_obj)*Alp.get_option(m, :rel_gap)
         else
            # if m.best_obj > 0
            #    cutoff_value = m.best_obj/(1.0 - Alp.get_option(m, :rel_gap))
            # else
            #    cutoff_value = m.best_obj/(1.0 + Alp.get_option(m, :rel_gap))
            # end
            cutoff_value = m.best_obj - abs(m.best_obj)*Alp.get_option(m, :rel_gap)
         end
         if m.mip_solver_id == "Gurobi"
            JuMP.set_optimizer_attribute(m.model_mip, "Cutoff", cutoff_value)
         elseif m.mip_solver_id == "Cplex"
            if Alp.is_min_sense(m)
               JuMP.set_optimizer_attribute(m.model_mip, "CPX_PARAM_CUTUP", cutoff_value)
            else
               JuMP.set_optimizer_attribute(m.model_mip, "CPX_PARAM_CUTLO", cutoff_value)
            end
         end
      end
   end

   # ADDED
   # if m.logs[:n_iter] == 1
   #    println("MIP model: ", m.model_mip)
   # end

   # ================= Solve Start ================ #
   start_bounding_solve = time()
   JuMP.optimize!(m.model_mip)
   status = JuMP.termination_status(m.model_mip)
   # m.logs[:total_time] += time() - start_bounding_solve
   # ADDED
   if add_time
      m.logs[:total_time] += time() - start_bounding_solve
   end
   m.logs[:time_left] = max(0.0, Alp.get_option(m, :time_limit) - m.logs[:total_time])
   # ================= Solve End ================ #

   if status in STATUS_OPT || status in STATUS_LIMIT

      # candidate_bound = (status == MOI.OPTIMAL) ? JuMP.objective_value(m.model_mip) : JuMP.objective_bound(m.model_mip)
      # ADDED
      candidate_bound = JuMP.objective_bound(m.model_mip)
      # candidate_bound_sol = [round.(JuMP.value(_index_to_variable_ref(m.model_mip, i)); digits = 7) 
      #                               for i in 1:(m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip)]
      # ADDED
      candidate_bound_sol = [JuMP.value(_index_to_variable_ref(m.model_mip, i))
                                    for i in 1:(m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip)]

      # Experimental code
      Alp.measure_relaxed_deviation(m, sol=candidate_bound_sol)
      if Alp.get_option(m, :disc_consecutive_forbid) > 0
         m.bound_sol_history[mod(m.logs[:n_iter]-1, Alp.get_option(m, :disc_consecutive_forbid))+1] = copy(candidate_bound_sol) # Requires proper offseting
      end
      push!(m.logs[:bound], candidate_bound)
      if eval(convertor[m.sense_orig])(candidate_bound, m.best_bound)
         m.best_bound = candidate_bound
         m.best_bound_sol = copy(candidate_bound_sol)
         m.status[:bounding_solve] = status
         m.detected_bound = true
      end

      # collect_lb_pool(m)    # Collect a pool of sub-optimal solutions - currently implemented for Gurobi only

   elseif status in STATUS_INF || status == MOI.INFEASIBLE_OR_UNBOUNDED

      push!(m.logs[:bound], "-")
      m.status[:bounding_solve] = MOI.INFEASIBLE
      @warn "  Warning: Infeasibility detected in the MIP solver"

      if ALPINE_DEBUG
         @warn "Use Alpine.print_iis_gurobi(m.model_mip) function in src/utility.jl (commented out code) for further investigation, if your MIP solver is Gurobi"
      end

   elseif status == :Unbounded

      m.status[:bounding_solve] = MOI.DUAL_INFEASIBLE
      @warn "  Warning: MIP solver returns unbounded"

   else

      error("  Warning: MIP solver failure $(status)")
   end
      
   # ADDED
   # println("Finished bounding solve \n")

   return
end


# ADDED function

"""
   bounding_solve_grad(m::Optimizer; kwargs...)

Obtain generalized gradient information for solving the max-min problem
"""
function bounding_solve_grad(m::Optimizer, mip_pointers::Dict, α_fix::Dict, discretization::Dict)

   λ = mip_pointers[:ext]
   α = mip_pointers[:part]
   con1 = mip_pointers[:x_con]
   con2 = mip_pointers[:aux_con]
   ml_indices = mip_pointers[:indices]
   sliced_indices = mip_pointers[:sliced_indices]

   con3 = mip_pointers[:mon_x_con]
   con4 = mip_pointers[:mon_aux_con]
   con5 = mip_pointers[:mon_part_con_low]
   con6 = mip_pointers[:mon_part_con_up]
   monomial_index = mip_pointers[:mon_indices]

   num_disc_points::Int64 = Alp.get_option(m, :disc_sp_num_points)
   genl_grad = Dict()
   len_disc = Dict()
   disc_points = Dict()
   x_orig_opt = Dict()
   for i in m.disc_vars
      genl_grad[i] = zeros(Float64,num_disc_points)
      len_disc[i] = length(discretization[i])
      disc_points[i] = deepcopy(discretization[i])
   end

   Alp.set_mip_time_limit(m)

   # Fix the partitioning variables to the active partition
   for k in keys(m.nonconvex_terms)
      nl_type = m.nonconvex_terms[k][:nonlinear_type]
      if nl_type == :BILINEAR
         for i in ml_indices[k]
            partition_cnt = length(discretization[i]) - 1
            JuMP.@constraint(m.model_mip, [j=1:partition_cnt], α[i][j] == α_fix[i][j])
         end
      elseif nl_type == :MONOMIAL
         i = monomial_index[k]
         partition_cnt = length(discretization[i]) - 1
         JuMP.@constraint(m.model_mip, [j=1:partition_cnt], α[i][j] == α_fix[i][j])
      end
   end

   # ================= Solve Start ================ #
   JuMP.relax_integrality(m.model_mip)
   
   JuMP.optimize!(m.model_mip)
   status = JuMP.termination_status(m.model_mip)
   # ================= Solve End ================ #


   if status in STATUS_OPT || status in STATUS_LIMIT

      for i in m.disc_vars
         x_orig_opt[i] = JuMP.value(_index_to_variable_ref(m.model_mip, i))
      end
      
      for k in keys(m.nonconvex_terms)
         nl_type = m.nonconvex_terms[k][:nonlinear_type]
         if nl_type == :BILINEAR
            λ_opt = JuMP.value.(λ[ml_indices[k]][:vars])
            dual_con2 = JuMP.dual(con2[ml_indices[k]])
            ind = Int64[0, 0]
            dual_con1 = Dict()

            for (cnt, i) in enumerate(ml_indices[k])
               ind[cnt] = i
               dual_con1[i] = JuMP.dual(con1[(ml_indices[k], i)])
            end

            sliced_indices_1 = sliced_indices[(ml_indices[k], ind[1])]
            sliced_indices_2 = sliced_indices[(ml_indices[k], ind[2])]

            if haskey(len_disc, ind[1]) && haskey(len_disc, ind[2])
               for j = 2:len_disc[ind[1]]-1
                  genl_grad[ind[1]][j-1] += sum(dual_con1[ind[1]]*λ_opt[sliced_indices_1[j][l]] + dual_con2*λ_opt[sliced_indices_1[j][l]]*discretization[ind[2]][l] for l in 1:len_disc[ind[2]])
               end
               for j = 2:len_disc[ind[2]]-1
                  genl_grad[ind[2]][j-1] += sum(dual_con1[ind[2]]*λ_opt[sliced_indices_2[j][l]] + dual_con2*λ_opt[sliced_indices_2[j][l]]*discretization[ind[1]][l] for l in 1:len_disc[ind[1]])
               end
            elseif haskey(len_disc, ind[1])
               for j = 2:len_disc[ind[1]]-1
                  genl_grad[ind[1]][j-1] += sum(dual_con1[ind[1]]*λ_opt[sliced_indices_1[j][l]] + dual_con2*λ_opt[sliced_indices_1[j][l]]*discretization[ind[2]][l] for l in 1:length(discretization[ind[2]]))
               end
            elseif haskey(len_disc, ind[2])
               for j = 2:len_disc[ind[2]]-1
                  genl_grad[ind[2]][j-1] += sum(dual_con1[ind[2]]*λ_opt[sliced_indices_2[j][l]] + dual_con2*λ_opt[sliced_indices_2[j][l]]*discretization[ind[1]][l] for l in 1:length(discretization[ind[1]]))
               end
            end


         elseif nl_type == :MULTILINEAR
            error("Do not support multilinear terms at the moment")


         elseif nl_type == :MONOMIAL
            λ_opt = JuMP.value.(λ[monomial_index[k]][:vars])
            dual_con3 = JuMP.dual(con3[monomial_index[k]])
            dual_con4 = JuMP.dual(con4[monomial_index[k]])
            dual_con5 = JuMP.dual(con5[monomial_index[k]])
            dual_con6 = JuMP.dual(con6[monomial_index[k]])
            ind = monomial_index[k]


            if haskey(len_disc, ind)
               for j = 2:len_disc[ind]-1
                  genl_grad[ind][j-1] += (dual_con3*λ_opt[j] + 2*dual_con4*λ_opt[j]*discretization[ind][j] + dual_con5*α_fix[ind][j] - dual_con6*α_fix[ind][j-1])
               end
            end
         end
      end

   elseif status in STATUS_INF || status == MOI.INFEASIBLE_OR_UNBOUNDED
      
      push!(m.logs[:bound], "-")
      # m.status[:bounding_solve] = MOI.INFEASIBLE
      @warn "  Warning: Infeasibility detected in the MIP solver during bounding_solve_grad"

      if ALPINE_DEBUG
         @warn "Use Alpine.print_iis_gurobi(m.model_mip) function in src/utility.jl (commented out code) for further investigation, if your MIP solver is Gurobi"
      end

   elseif status == :Unbounded
      
      # m.status[:bounding_solve] = MOI.DUAL_INFEASIBLE
      @warn "  Warning: MIP solver returns unbounded"

   else

      error("  Warning: MIP solver failure $(status)")
   end

   return genl_grad, status
end


"""
   pick_disc_vars(m::Optimizer)

This function helps pick the variables for discretization. The method chosen depends on user-inputs.
In case when `indices::Int` is provided, the method is chosen as built-in method. Currently,
there are two built-in options for users as follows:

* `max_cover (Alp.get_option(m, :disc_var_pick)=0, default)`: pick all variables involved in the non-linear term for discretization
* `min_vertex_cover (Alp.get_option(m, :disc_var_pick)=1)`: pick a minimum vertex cover for variables involved in non-linear terms so that each non-linear term is at least convexified

For advanced usage, `Alp.get_option(m, :disc_var_pick)` allows `::Function` inputs. User can provide his/her own function to choose the variables for discretization.

"""
function pick_disc_vars(m::Optimizer)

   disc_var_pick = Alp.get_option(m, :disc_var_pick)

   if isa(disc_var_pick, Function)
      # eval(Alp.get_option(m, :disc_var_pick))(m)
      disc_var_pick(m)
      length(m.disc_vars) == 0 && length(m.nonconvex_terms) > 0 && error("[USER FUNCTION] must select at least one variable to perform discretization for convexificiation purpose")
   elseif isa(disc_var_pick, Int)
      if disc_var_pick == 0
         Alp.get_candidate_disc_vars(m)
      elseif disc_var_pick == 1
         Alp.min_vertex_cover(m)
      elseif disc_var_pick == 2
         (length(m.candidate_disc_vars) > 15) ? Alp.min_vertex_cover(m) : Alp.get_candidate_disc_vars(m)
      elseif disc_var_pick == 3 # Initial
         (length(m.candidate_disc_vars) > 15) ? Alp.min_vertex_cover(m) : Alp.get_candidate_disc_vars(m)
      else
         error("Unsupported default indicator for picking variables for discretization")
      end
   else
      error("Input for parameter :disc_var_pick is illegal. Should be either a Int for default methods indexes or functional inputs.")
   end

   return
end
