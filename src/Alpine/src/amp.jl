"""
    create_bounding_mip(m::Optimizer; use_disc::Dict)

Set up a MILP bounding model base on variable domain partitioning information stored in `use_disc`.
By default, if `use_disc` is not provided, it will use `m.discretizations` store in the Alpine model.
The basic idea of this MILP bounding model is to use Tighten McCormick to convexify the original Non-convex region.
Among all presented partitions, the bounding model will choose one specific partition as the lower bound solution.
The more partitions there are, the better or finer bounding model relax the original MINLP while the more
efforts required to solve this MILP is required.

This function is implemented in the following manner:

    * [`amp_post_vars`](@ref): post original and lifted variables
    * [`amp_post_lifted_constraints`](@ref): post original and lifted constraints
    * [`amp_post_lifted_obj`](@ref): post original or lifted objective function
    * [`amp_post_tmc_mccormick`](@ref): post Tighten McCormick variables and constraints base on `discretization` information
"""
# function create_bounding_mip(m::Optimizer; use_disc=nothing)
# ADDED
function create_bounding_mip(m::Optimizer; kwargs...)
#=
    use_disc === nothing ? discretization = m.discretization : discretization = use_disc

    m.model_mip = Model(Alp.get_option(m, :mip_solver)) # Construct JuMP Model
    start_build = time()
    # ------- Model Construction ------ #
    Alp.amp_post_vars(m)                                                # Post original and lifted variables
    Alp.amp_post_lifted_constraints(m)                                  # Post lifted constraints
    Alp.amp_post_lifted_objective(m)                                    # Post objective
    Alp.amp_post_convexification(m, use_disc=discretization)            # Convexify problem
    # --------------------------------- #
    cputime_build = time() - start_build
    m.logs[:total_time] += cputime_build
    m.logs[:time_left] = max(0.0, Alp.get_option(m, :time_limit) - m.logs[:total_time])

    return
=#
    # ADDED
    options = Dict(kwargs)

    haskey(options, :use_disc) ? discretization = options[:use_disc] : discretization = m.discretization
    haskey(options, :add_time) ? add_time = options[:add_time] : add_time = true

    m.model_mip = Model(Alp.get_option(m, :mip_solver)) # Construct JuMP Model
    start_build = time()
    # ------- Model Construction ------ #
    if m.logs[:n_iter] >= Alp.get_option(m, :disc_sp_iter) && m.logs[:n_iter] > 0
        Alp.amp_post_vars(m, use_disc = discretization)                      # Post original and lifted variables
    else
        Alp.amp_post_vars(m)                                                 # Post original and lifted variables
    end
    Alp.amp_post_lifted_constraints(m)                                       # Post lifted constraints
    Alp.amp_post_lifted_objective(m)                                         # Post objective
    mip_pointers = Alp.amp_post_convexification(m, use_disc=discretization)  # Convexify problem
    # --------------------------------- #
    cputime_build = time() - start_build
    if add_time
        m.logs[:total_time] += cputime_build
    end
    m.logs[:time_left] = max(0.0, Alp.get_option(m, :time_limit) - m.logs[:total_time])

    return mip_pointers
end

"""
    amp_post_convexification(m::Optimizer; kwargs...)

wrapper function to convexify the problem for a bounding model. This function talks to nonconvex_terms and convexification methods
to finish the last step required during the construction of bounding model.
"""
function amp_post_convexification(m::Optimizer; use_disc=nothing)

    use_disc === nothing ? discretization = m.discretization : discretization = use_disc

    # for i in 1:length(Alp.get_option(m, :method_convexification))             # Additional user-defined convexification method
    #    eval(Alp.get_option(m, :method_convexification)[i])(m)
    #    Alp.get_option(m, :method_convexification)[i](m)
    # end

    Alp.amp_post_mccormick(m, use_disc=discretization)            # handles all bi-linear and monomial convexificaitons
    # Alp.amp_post_convhull(m, use_disc=discretization)           # convex hull representation
    # ADDED return
    mip_pointers = Alp.amp_post_convhull(m, use_disc=discretization)           # convex hull representation

    Alp.is_fully_convexified(m) # Ensure if  all the non-linear terms are convexified

    # return
    return mip_pointers
end

function amp_post_vars(m::Optimizer; kwargs...)

    options = Dict(kwargs)

    if haskey(options, :use_disc)
        l_var = [options[:use_disc][i][1]   for i in 1:(m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip)]
        u_var = [options[:use_disc][i][end] for i in 1:(m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip)]
    else
        l_var = m.l_var_tight
        u_var = m.u_var_tight
    end

    JuMP.@variable(m.model_mip, x[i=1:(m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip)])

    for i in 1:(m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip)
        # Interestingly, not enforcing category of lifted variables is able to improve performance
        if i <= m.num_var_orig
            if m.var_type_orig[i] == :Bin
                set_binary(x[i])
            elseif m.var_type_orig[i] == :Int
                set_integer(x[i])
            end
        end
        # Changed to tight bound, if no bound tightening is performed, will be just .l_var_orig
        l_var[i] > -Inf && JuMP.set_lower_bound(x[i], l_var[i])
        # Changed to tight bound, if no bound tightening is performed, will be just .u_var_orig
        u_var[i] < Inf && JuMP.set_upper_bound(x[i], u_var[i])

        m.var_type[i] == :Int && error("Alpine does not support MINLPs with generic integer (non-binary) variables yet! Try Juniper.jl for finding a local feasible solution")
    end

    return
end


function amp_post_lifted_constraints(m::Optimizer)

    for i in 1:m.num_constr_orig
        if m.constr_structure[i] == :affine
            Alp.amp_post_affine_constraint(m.model_mip, m.bounding_constr_mip[i])
        elseif m.constr_structure[i] == :convex
            Alp.amp_post_convex_constraint(m.model_mip, m.bounding_constr_mip[i])
        else
            error("Unknown constr_structure type $(m.constr_structure[i])")
        end
    end

    for i in keys(m.linear_terms)
        Alp.amp_post_linear_lift_constraints(m.model_mip, m.linear_terms[i])
    end

    return
end

function amp_post_affine_constraint(model_mip::JuMP.Model, affine::Dict)

    if affine[:sense] == :(>=)
        JuMP.@constraint(model_mip,
            sum(affine[:coefs][j]*_index_to_variable_ref(model_mip, affine[:vars][j].args[2]) for j in 1:affine[:cnt]) >= affine[:rhs])
    elseif affine[:sense] == :(<=)
        JuMP.@constraint(model_mip,
            sum(affine[:coefs][j]*_index_to_variable_ref(model_mip, affine[:vars][j].args[2]) for j in 1:affine[:cnt]) <= affine[:rhs])
    elseif affine[:sense] == :(==)
        JuMP.@constraint(model_mip,
            sum(affine[:coefs][j]*_index_to_variable_ref(model_mip, affine[:vars][j].args[2]) for j in 1:affine[:cnt]) == affine[:rhs])
    else
        error("Unkown sense.")
    end

    return
end

function amp_post_convex_constraint(model_mip::JuMP.Model, convex::Dict)

    !prod([i == 2 for i in convex[:powers]]) && error("No relaxation implementation for convex constraints $(convex[:expr])")

    if convex[:sense] == :(<=)
        JuMP.@constraint(model_mip,
            sum(convex[:coefs][j]*_index_to_variable_ref(model_mip, convex[:vars][j].args[2])^2 for j in 1:convex[:cnt]) <= convex[:rhs])
    elseif convex[:sense] == :(>=)
        JuMP.@constraint(model_mip,
            sum(convex[:coefs][j]*_index_to_variable_ref(model_mip, convex[:vars][j].args[2])^2 for j in 1:convex[:cnt]) >= convex[:rhs])
    else
        error("No equality constraints should be recognized as supported convex constriants")
    end

    return
end

function amp_post_linear_lift_constraints(model_mip::JuMP.Model, l::Dict)

    @assert l[:ref][:sign] == :+
    JuMP.@constraint(model_mip, _index_to_variable_ref(model_mip, l[:y_idx]) == sum(i[1]*_index_to_variable_ref(model_mip, i[2]) for i in l[:ref][:coef_var]) + l[:ref][:scalar])
    return
end

function amp_post_lifted_objective(m::Optimizer)

#if isa(m.obj_expr_orig, Number)
if expr_isconst(m.obj_expr_orig)
    JuMP.@objective(m.model_mip, m.sense_orig, eval(m.obj_expr_orig))
   elseif m.obj_structure == :affine
        JuMP.@objective(m.model_mip, m.sense_orig, m.bounding_obj_mip[:rhs] + sum(m.bounding_obj_mip[:coefs][i]*_index_to_variable_ref(m.model_mip, m.bounding_obj_mip[:vars][i].args[2]) for i in 1:m.bounding_obj_mip[:cnt]))
    elseif m.obj_structure == :convex
        # This works only when the original objective is convex quadratic.
        # Higher-order convex monomials need implementation of outer-approximation (check resolve_convex_constr in operators.jl)
        JuMP.@objective(m.model_mip, m.sense_orig, m.bounding_obj_mip[:rhs] + sum(m.bounding_obj_mip[:coefs][i]*_index_to_variable_ref(m.model_mip, m.bounding_obj_mip[:vars][i].args[2])^2 for i in 1:m.bounding_obj_mip[:cnt]))
    else
        error("Unknown structural obj type $(m.obj_structure)")
    end
    return
end

function add_partition(m::Optimizer; kwargs...)

    options = Dict(kwargs)
    haskey(options, :use_disc) ? discretization = options[:use_disc] : discretization = m.discretization
    haskey(options, :use_solution) ? point_vec = options[:use_solution] : point_vec = m.best_bound_sol
#=
    if isa(Alp.get_option(m, :disc_add_partition_method), Function)
        # m.discretization = eval(Alp.get_option(m, :disc_add_partition_method))(m, use_disc=discretization, use_solution=point_vec)
        m.discretization = Alp.get_option(m, :disc_add_partition_method)(m, use_disc=discretization, use_solution=point_vec)
    elseif Alp.get_option(m, :disc_add_partition_method) == "adaptive"
        m.discretization = Alp.add_adaptive_partition(m, use_disc=discretization, use_solution=point_vec)
    elseif Alp.get_option(m, :disc_add_partition_method) == "uniform"
        m.discretization = add_uniform_partition(m, use_disc=discretization)
    else
        error("Unknown input on how to add partitions.")
    end
=#
    # ADDED
    if isa(Alp.get_option(m, :disc_add_partition_method), Function)
        # m.discretization = eval(Alp.get_option(m, :disc_add_partition_method))(m, use_disc=discretization, use_solution=point_vec)
        m.discretization = Alp.get_option(m, :disc_add_partition_method)(m, use_disc=discretization, use_solution=point_vec)
    elseif Alp.get_option(m, :disc_add_partition_method) == "adaptive"
        if m.logs[:n_iter] >= Alp.get_option(m, :disc_sp_iter)
            if Alp.get_option(m, :disc_use_single_partitions)
                if m.logs[:n_iter] > 0
                    m.discretization = Alp.add_single_partition(m, use_disc=discretization, use_solution=point_vec)
                else
                    # compute the McCormick solution (without any partitions) for the first iteration and use it as the partitioning point
                    init_best_bound = m.best_bound
                    Alp.create_bounding_mip(m, use_disc=discretization)
                    Alp.bounding_solve(m)
                    m.best_bound = init_best_bound

                    m.discretization = Alp.add_single_partition(m, use_disc=discretization, use_solution=m.best_bound_sol)
                end
            else
                m.discretization = Alp.add_adaptive_partition(m, use_disc=discretization, use_solution=point_vec)
            end
        elseif Alp.get_option(m, :disc_use_specified_part)
            disc_specified_part = Alp.get_option(m, :disc_specified_part)
            if m.logs[:n_iter] == 0
                disc_not_specified = false
                disc_not_specified_iter = []
                for j = 1:Alp.get_option(m, :disc_sp_iter)
                    if !haskey(disc_specified_part, string(j))
                        disc_not_specified = true
                        push!(disc_not_specified_iter, j)
                    end
                end
                if disc_not_specified
                    error("Partitioning points not specified for iterations ",disc_not_specified_iter)
                end
            end

            curr_iter::Int = m.logs[:n_iter]+1
            m.discretization = Alp.add_specified_partitions(m, discretization, disc_specified_part[string(curr_iter)])
        else
            if m.logs[:n_iter] > 0
                m.discretization = Alp.add_sp_partition(m, use_disc=discretization, use_solution=point_vec)
            else
                # compute the McCormick solution (without any partitions) for the first iteration and use it to inform the partitioning point
                init_best_bound = m.best_bound
                Alp.create_bounding_mip(m, use_disc=discretization, add_time=false)
                Alp.bounding_solve(m, add_time=false)
                m.best_bound = init_best_bound

                m.discretization = Alp.add_sp_partition(m, use_disc=discretization, use_solution=m.best_bound_sol)
            end
        end
    elseif Alp.get_option(m, :disc_add_partition_method) == "uniform"
        m.discretization = Alp.add_uniform_partition(m, use_disc=discretization)
    else
        error("Unknown input on how to add partitions.")
    end

    return
end

# TODO: also need to document the special diverted cases when new partition touches both corners
"""

    add_adaptive_partition(m::Optimizer; use_disc::Dict, use_solution::Vector)

A built-in method used to add a new partition on feasible domains of variables chosen for partitioning.

This can be illustrated by the following example. Let the previous iteration's partition vector on 
variable "x" be given by [0, 3, 7, 9]. And say, the lower bounding solution has a value of 4 for variable "x".
In the case when `disc_ratio=4`, this function creates the new partition vector as follows: [0, 3, 3.5, 4, 4.5, 7, 9]

There are two options for this function,

    * `use_disc(default=m.discretization)`:: to regulate which is the base to add new partitions on
    * `use_solution(default=m.best_bound_sol)`:: to regulate which solution to use when adding new partitions on

This function can be accordingly modified by the user to change the behavior of the solver, and thus the convergence.

"""
function add_adaptive_partition(m::Optimizer;kwargs...)

    # ADDED
    # start_time_adaptive = time()

    options = Dict(kwargs)

    haskey(options, :use_disc) ? discretization = options[:use_disc] : discretization = m.discretization
    haskey(options, :use_solution) ? point_vec = copy(options[:use_solution]) : point_vec = copy(m.best_bound_sol)
    haskey(options, :use_ratio) ? ratio = options[:use_ratio] : ratio = Alp.get_option(m, :disc_ratio)
    haskey(options, :branching) ? branching = options[:branching] : branching = false

    # ADDED: use input discretization ratio for iteration 1
    if m.logs[:n_iter] == 0
        if Alp.get_option(m, :disc_ratio_iter1) > 1
            ratio = Alp.get_option(m, :disc_ratio_iter1)
            println("Using discretization ratio ", ratio," during presolve")
        end
    end
    
    if length(point_vec) < m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip
        point_vec = Alp.resolve_lifted_var_value(m, point_vec)  # Update the solution vector for lifted variable
    end
    
    if branching
        discretization = deepcopy(discretization)
    end

    processed = Set{Int}()

    # Perform discretization based on type of nonlinear terms #
    # ADDED
    init_discretization = deepcopy(discretization)
    # println("Current discretization: ")

    for i in m.disc_vars
        point = point_vec[i]                # Original Variable
        λCnt = length(discretization[i])

        # Built-in method based-on variable type
        if m.var_type[i] == :Cont
            i in processed && continue
            point = Alp.correct_point(m, discretization[i], point, i)
            for j in 1:λCnt
                if point >= discretization[i][j] && point <= discretization[i][j+1]  # Locating the right location
                    radius = Alp.calculate_radius(discretization[i], j, ratio)
                    Alp.insert_partition(m, i, j, point, radius, discretization[i])
                    push!(processed, i)
                    break
                end
            end
        elseif m.var_type[i] == :Bin # This should never happen
            @warn "  Warning: Binary variable in m.disc_vars. Check out what is wrong..."
            continue  # No partition should be added to binary variable unless user specified
        elseif m.var_type[i] == :Int
            error("Alpine does not support MINLPs with generic integer (non-binary) variables yet!")
        else
            error("Unexpected variable types while inserting partitions")
        end

        # ADDED
        if m.logs[:n_iter] == 0
            println("disc:  ",i,"  :",discretization[i])
        end
    end

    # ADDED
    if Alp.check_partition_stalling(m,init_discretization,discretization)
        Alp.create_bounding_mip(m, use_disc=discretization)
        Alp.bounding_solve(m)
        m.status[:bounding_solve] = MOI.NUMERICAL_ERROR
        @warn "  Warning: Alpine is terminating because the variable partitions did not change in the current iteration. Problem may be badly scaled."
    end

    # ADDED
    # if m.logs[:n_iter] == 0
    #     println("Time for setting up partitions in presolve: ", time() - start_time_adaptive)
    # end

    return discretization
end

"""
    This function targets to address unexpected numerical issues when adding partitions in tight regions.
"""
function correct_point(m::Optimizer, partvec::Vector, point::Float64, var::Int)

    tol = Alp.get_option(m, :tol)

    if point < partvec[1] - tol || point > partvec[end] + tol
        @warn "  Warning: VAR$(var) SOL=$(point) out of discretization [$(partvec[1]),$(partvec[end])]. Hence, taking the middle point."
        return 0.5*(partvec[1] + partvec[end]) # Should choose the longest range
    end

    isapprox(point, partvec[1];   atol = tol) && return partvec[1]
    isapprox(point, partvec[end]; atol = tol) && return partvec[end]

    return point
end

function calculate_radius(partvec::Vector, part::Int, ratio::Any)

    lb_local = partvec[part]
    ub_local = partvec[part+1]

    distance = ub_local - lb_local
    if isa(ratio, Float64) || isa(ratio, Int)
        radius = distance / ratio
    elseif isa(ratio, Function)
        radius = distance / ratio(m)
    else
        error("Undetermined discretization ratio")
    end

    return radius
end

function insert_partition(m::Optimizer, var::Int, partidx::Int, point::Number, radius::Float64, partvec::Vector)

    abstol, reltol = Alp.get_option(m, :disc_abs_width_tol), Alp.get_option(m, :disc_rel_width_tol)

    lb_local, ub_local = partvec[partidx], partvec[partidx+1]
    ub_touch, lb_touch = true, true
    lb_new, ub_new = max(point - radius, lb_local), min(point + radius, ub_local)

    if (!isapprox(point,ub_local; atol=abstol) && !isapprox(point,lb_local; atol=abstol)) || Alp.get_option(m, :disc_partition_at_existing)
        if ub_new < ub_local && !isapprox(ub_new, ub_local; atol=abstol) && abs(ub_new-ub_local)/(1e-8+abs(ub_local)) > reltol # Insert new UB-based partition
            insert!(partvec, partidx+1, ub_new)
            ub_touch = false
        end

        if lb_new > lb_local && !isapprox(lb_new, lb_local; atol=abstol) && abs(lb_new-lb_local)/(1e-8+abs(lb_local)) > reltol # Insert new LB-based partition
            insert!(partvec, partidx+1, lb_new)
            lb_touch = false
        end
    end
#=
    if (ub_touch && lb_touch) || Alp.check_solution_history(m, var)
        distvec = [(j, partvec[j+1]-partvec[j]) for j in 1:length(partvec)-1]
        sort!(distvec, by=x->x[2])
        point_orig = point
        pos = distvec[end][1]
        lb_local = partvec[pos]
        ub_local = partvec[pos+1]
        isapprox(lb_local, ub_local; atol = Alp.get_option(m, :tol)) && return
        chunk = (ub_local - lb_local) / Alp.get_option(m, :disc_divert_chunks)
        point = lb_local + (ub_local - lb_local) / Alp.get_option(m, :disc_divert_chunks)
        for i in 2:Alp.get_option(m, :disc_divert_chunks)
            insert!(partvec, pos+1, lb_local + chunk * (Alp.get_option(m, :disc_divert_chunks)-(i-1)))
        end
        (Alp.get_option(m, :log_level) > 199) && println("[DEBUG] !D! VAR$(var): SOL=$(round(point_orig; digits = 4))=>$(point) |$(round(lb_local; digits = 4)) | $(Alp.get_option(m, :disc_divert_chunks)) SEGMENTS | $(round(ub_local; digits = 4))|")
=#
    # ADDED
    if (ub_touch && lb_touch) || Alp.check_solution_history(m, var)
        # do nothing
    else
        (Alp.get_option(m, :log_level) > 199) && println("[DEBUG] VAR$(var): SOL=$(round(point; digits = 4)) RADIUS=$(radius), PARTITIONS=$(length(partvec)-1) |$(round(lb_local; digits = 4)) |$(round(lb_new; digits = 6)) <- * -> $(round(ub_new; digits = 6))| $(round(ub_local; digits = 4))|")
    end

    return
end

function add_uniform_partition(m::Optimizer; kwargs...)

    options = Dict(kwargs)
    haskey(options, :use_disc) ? discretization = options[:use_disc] : discretization = m.discretization

    for i in m.disc_vars  # Only construct when discretized
        lb_local = discretization[i][1]
        ub_local = discretization[i][end]
        distance = ub_local - lb_local
        chunk = distance / ((m.logs[:n_iter]+1)*Alp.get_option(m, :disc_uniform_rate))
        discretization[i] = [lb_local+chunk*(j-1) for j in 1:(m.logs[:n_iter]+1)*Alp.get_option(m, :disc_uniform_rate)]
        push!(discretization[i], ub_local)   # Safety Scheme
        (Alp.get_option(m, :log_level) > 199) && println("[DEBUG] VAR$(i): RATE=$(Alp.get_option(m, :disc_uniform_rate)), PARTITIONS=$(length(discretization[i]))  |$(round(lb_local; digits=4)) | $(Alp.get_option(m, :disc_uniform_rate)*(1+m.logs[:n_iter])) SEGMENTS | $(round(ub_local; digits=4))|")
    end

    return discretization
end

function update_disc_ratio(m::Optimizer, presolve=false)

    m.logs[:n_iter] > 2 && return Alp.get_option(m, :disc_ratio) # Stop branching after the second iterations

    ratio_pool = [8:2:20;]  # Built-in try range
    convertor = Dict(MOI.MAX_SENSE => :<, MOI.MIN_SENSE => :>)
    # revconvertor = Dict(MOI.MAX_SENSE => :>, MOI.MIN_SENSE => :<)

    incumb_ratio = ratio_pool[1]
    Alp.is_min_sense(m) ? incumb_res = -Inf : incumb_res = Inf
    res_collector = Float64[]

    for r in ratio_pool
        st = time()
        if !isempty(m.best_sol)
            branch_disc = Alp.add_adaptive_partition(m, use_disc = m.discretization, branching = true, use_ratio = r, use_solution = m.best_sol)
        else
            branch_disc = Alp.add_adaptive_partition(m, use_disc = m.discretization, branching = true, use_ratio = r)
        end
        Alp.create_bounding_mip(m, use_disc = branch_disc)
        res = Alp.disc_branch_solve(m)
        push!(res_collector, res)
        if eval(convertor[m.sense_orig])(res, incumb_res) # && abs(abs(collector[end]-res)/collector[end]) > 1e-1  # %1 of difference
            incumb_res = res
            incumb_ratio = r
        end
        et = time() - st
        if et > 300  # 5 minutes limitation
            println("Expensive disc branching pass... Fixed at 8")
            return 8
        end
        Alp.get_option(m, :log_level) > 0 && println("BRANCH RATIO = $(r), METRIC = $(res) || TIME = $(time()-st)")
    end

    if Statistics.std(res_collector) >= 1e-2    # Detect if all solutions are similar to each other
        Alp.get_option(m, :log_level) > 0 && println("RATIO BRANCHING OFF due to solution variance test passed.")
        Alp.set_option(m, :disc_ratio_branch, false) # If an incumbent ratio is selected, then stop the branching scheme
    end

    if !isempty(m.best_sol)
        m.discretization = Alp.add_adaptive_partition(m, use_disc = m.discretization, branching = true, use_ratio = incumb_ratio, use_solution = m.best_sol)
    else
        m.discretization = Alp.add_adaptive_partition(m, use_disc = m.discretization, branching = true, use_ratio = incumb_ratio)
    end

    Alp.get_option(m, :log_level) > 0 && println("INCUMB_RATIO = $(incumb_ratio)")

    return incumb_ratio
end

function disc_branch_solve(m::Optimizer)

    # ================= Solve Start ================ #
    Alp.set_mip_time_limit(m)
    start_bounding_solve = time()
    JuMP.optimize!(m.model_mip)
    status = MOI.get(m.model_mip, MOI.TerminationStatus())
    cputime_branch_bounding_solve = time() - start_bounding_solve
    m.logs[:total_time] += cputime_branch_bounding_solve
    m.logs[:time_left] = max(0.0, Alp.get_option(m, :time_limit) - m.logs[:total_time])
    # ================= Solve End ================ #

    if status in STATUS_OPT || status in STATUS_LIMIT
        return MOI.get(m.model_mip, MOI.ObjectiveBound())
    else
        @warn "  Warning: Unexpected solving condition $(status) during disc branching."
    end

    # Safety scheme
    if Alp.is_min_sense(m)
        return -Inf
    elseif Alp.is_max_sense(m)
        return Inf
    end
end


# ADDED new methods

"""
Method to check if Alpine's discretizations are stalling
"""
function check_partition_stalling(m::Optimizer,disc1::Dict{Any,Any},disc2::Dict{Any,Any})

    for i in m.disc_vars
        if disc1[i] != disc2[i]
            return false
        end
    end

    return true
end



"""
Method to add single partitions at the bounding solution during each iteration of Alpine
"""
function add_single_partition(m::Optimizer;kwargs...)

    options = Dict(kwargs)

    haskey(options, :use_disc) ? discretization = options[:use_disc] : discretization = m.discretization
    haskey(options, :use_solution) ? point_vec = copy(options[:use_solution]) : point_vec = copy(m.best_bound_sol)
    haskey(options, :use_ratio) ? ratio = options[:use_ratio] : ratio = Alp.get_option(m, :disc_ratio)
    haskey(options, :branching) ? branching = options[:branching] : branching = false

    if length(point_vec) < m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip
        point_vec = Alp.resolve_lifted_var_value(m, point_vec)
    end

    if branching
        discretization = deepcopy(discretization)
    end

    processed = Set{Int}()

    for i in m.disc_vars
        point = point_vec[i]
        λCnt = length(discretization[i])

        if m.var_type[i] == :Cont
            i in processed && continue
            point = Alp.correct_point(m, discretization[i], point, i)
            for j in 1:λCnt
                if point >= discretization[i][j] && point <= discretization[i][j+1]
                    Alp.insert_single_partition(m, j, point, discretization[i])
                    push!(processed, i)
                    break
                end
            end
        elseif m.var_type[i] == :Bin # This should never happen
            @warn "  Warning: Binary variable in m.disc_vars. Check out what is wrong..."
            continue
        elseif m.var_type[i] == :Int
            error("Alpine does not support MINLPs with generic integer (non-binary) variables yet!")
        else
            error("Unexpected variable types while inserting partitions")
        end
    end

    return discretization
end

"""
Method to insert a single partition for the chosen variable at the specified point
"""
function insert_single_partition(m::Optimizer, partidx::Int, point::Number, partvec::Vector)

    abstol, reltol = Alp.get_option(m, :disc_abs_width_tol), Alp.get_option(m, :disc_rel_width_tol)

    lb_local, ub_local = partvec[partidx], partvec[partidx+1]
    lb_touch, ub_touch = true, true

    # same checks as in insert_partition
    if point < ub_local && !isapprox(point, ub_local; atol=abstol) && abs(point-ub_local)/(1e-8+abs(ub_local)) > reltol
        ub_touch = false
    end

    if point > lb_local && !isapprox(point, lb_local; atol=abstol) && abs(point-lb_local)/(1e-8+abs(lb_local)) > reltol 
        lb_touch = false
    end

    if (!lb_touch) && (!ub_touch)
        insert!(partvec, partidx+1, point)
    end

    return
end

"""
Method to add partitions by solving the max_min problem
"""
function add_sp_partition(m::Optimizer;kwargs...)

    start_max_min = time()

    options = Dict(kwargs)

    haskey(options, :use_disc) ? discretization = options[:use_disc] : discretization = m.discretization
    haskey(options, :use_solution) ? point_vec = copy(options[:use_solution]) : point_vec = copy(m.best_bound_sol)

    tmp_discretization = deepcopy(discretization)

    disc_lower_bound, disc_upper_bound = Alp.compute_sp_bounds(m, discretization, use_solution=point_vec)
    for (var_count, i) in enumerate(m.disc_vars)
        tmp_discretization[i] = [disc_lower_bound[var_count], disc_upper_bound[var_count]]
    end

    if Alp.get_option(m, :disc_sp_method) == "mpbngc"
        for k in keys(m.nonconvex_terms)
            nl_type = m.nonconvex_terms[k][:nonlinear_type]
            if (!(nl_type == :BILINEAR || nl_type == :MULTILINEAR || nl_type == :MONOMIAL))
                println("nl_type: ", nl_type)
                error("Strong Partitioning with MPBNGC is only supported for monomial, bilinear, and multilinear terms at the moment.")
            end
        end
    end

    disc_points_sp = Alp.compute_sp_partition(m, use_disc=tmp_discretization, use_solution=point_vec)

    num_disc_points::Int = Alp.get_option(m, :disc_sp_num_points)

    for (var_count, i) in enumerate(m.disc_vars)
        disc_points_sp_sorted = sort(disc_points_sp[(var_count-1)*num_disc_points+1:var_count*num_disc_points])
        # Add discretization points one at a time so that no two discretization points are too close
        # Not the fastest implementation, but will do for now...
        for k = 1:num_disc_points
            λCnt = length(discretization[i])
            point = Alp.correct_point(m, discretization[i], disc_points_sp_sorted[k], i)
            for j in 1:λCnt
                if point >= discretization[i][j] && point <= discretization[i][j+1]
                    Alp.insert_single_partition(m, j, point, discretization[i])
                    break
                end
            end
        end
    end

    cputime_max_min = time() - start_max_min
    m.logs[:max_min_time] += cputime_max_min

    return discretization
end

"""
Compute bounds for strong partitioning points
"""
function compute_sp_bounds(m::Optimizer,discretization::Dict;kwargs...)

    options = Dict(kwargs)
    haskey(options, :use_solution) ? point_vec = copy(options[:use_solution]) : point_vec = copy(m.best_bound_sol)

    num_disc_vars::Int = length(m.disc_vars)
    disc_lower_bound = zeros(Float64,num_disc_vars)
    disc_upper_bound = zeros(Float64,num_disc_vars)

    processed = Set{Int}()

    for (var_count, i) in enumerate(m.disc_vars)
        point = point_vec[i]
        λCnt = length(discretization[i])

        if m.var_type[i] == :Cont
            i in processed && continue
            point = Alp.correct_point(m, discretization[i], point, i)
            for j in 1:λCnt
                if point >= discretization[i][j] && point <= discretization[i][j+1]
                    disc_lower_bound[var_count] = discretization[i][j]
                    disc_upper_bound[var_count] = discretization[i][j+1]
                    push!(processed, i)
                    break
                end
            end
        elseif m.var_type[i] == :Bin # This should never happen
            @warn "  Warning: Binary variable in m.disc_vars. Check out what is wrong..."
            continue
        elseif m.var_type[i] == :Int
            error("Alpine does not support MINLPs with generic integer (non-binary) variables yet!")
        else
            error("Unexpected variable types while inserting partitions")
        end
    end

    return disc_lower_bound, disc_upper_bound
end

"""
Compute initial guess for strong partitioning points
"""
function compute_sp_initial_guess(m::Optimizer,discretization::Dict,point_vec::Vector{Float64})

	init_discretization = deepcopy(discretization)
	init_best_bound = m.best_bound

    num_disc_vars::Int = length(m.disc_vars)
    num_disc_points::Int = Alp.get_option(m, :disc_sp_num_points)
    abstol, reltol = Alp.get_option(m, :disc_abs_width_tol), Alp.get_option(m, :disc_rel_width_tol)

    disc_start = zeros(Float64,num_disc_points*num_disc_vars)
    disc_lower_bound = zeros(Float64,num_disc_points*num_disc_vars)
    disc_upper_bound = zeros(Float64,num_disc_points*num_disc_vars)
    fix_disc_point = zeros(Int64,num_disc_points*num_disc_vars)

    # set bounds for the discretization points
    for (var_count, i) in enumerate(m.disc_vars)
        λCnt = length(init_discretization[i])
        disc_lower_bound[(var_count-1)*num_disc_points+1:var_count*num_disc_points] = (init_discretization[i][1])*ones(Float64,num_disc_points)
        disc_upper_bound[(var_count-1)*num_disc_points+1:var_count*num_disc_points] = (init_discretization[i][λCnt])*ones(Float64,num_disc_points)
    end

    if !Alp.get_option(m, :disc_use_initial_guess)

        # Set initial guess to be iterative single partition points
        for (var_count, i) in enumerate(m.disc_vars)
            bounds_touch = false    
            if Alp.get_option(m, :disc_sp_adaptive_part)
                ub_local = disc_upper_bound[(var_count-1)*num_disc_points+1]
                if point_vec[i] >= ub_local || isapprox(point_vec[i], ub_local; atol=abstol) || abs(point_vec[i]-ub_local)/(1e-8+abs(ub_local)) <= reltol
                    bounds_touch = true
                end
                lb_local = disc_lower_bound[(var_count-1)*num_disc_points+1]
                if point_vec[i] <= lb_local || isapprox(point_vec[i], lb_local; atol=abstol) || abs(point_vec[i]-lb_local)/(1e-8+abs(lb_local)) <= reltol
                    bounds_touch = true
                end
            end

            if bounds_touch
                disc_start[(var_count-1)*num_disc_points+1] = lb_local
                fix_disc_point[(var_count-1)*num_disc_points+1] = 1
            else
                disc_start[(var_count-1)*num_disc_points+1] = point_vec[i]
            end

            # println("point_vec[",i,"]: ",point_vec[i],"  fix_disc_point: ",(fix_disc_point[(var_count-1)*num_disc_points+1] > 0.5))
        end


        point_vec2 = deepcopy(point_vec)

        for j = 2:num_disc_points
            tmp_discretization = Alp.add_single_partition(m, use_disc=discretization, use_solution=point_vec2)
            Alp.create_bounding_mip(m, use_disc=tmp_discretization, add_time=false)
            Alp.bounding_solve(m, add_time=false)
            point_vec2 = deepcopy(m.best_bound_sol)
            for (var_count, i) in enumerate(m.disc_vars)
                bounds_touch2 = false    
                if Alp.get_option(m, :disc_sp_adaptive_part)
                    ub_local2 = disc_upper_bound[(var_count-1)*num_disc_points+1]
                    if point_vec2[i] >= ub_local2 || isapprox(point_vec2[i], ub_local2; atol=abstol) || abs(point_vec2[i]-ub_local2)/(1e-8+abs(ub_local2)) <= reltol
                        bounds_touch2 = true
                    end
                    lb_local2 = disc_lower_bound[(var_count-1)*num_disc_points+1]
                    if point_vec2[i] <= lb_local2 || isapprox(point_vec2[i], lb_local2; atol=abstol) || abs(point_vec2[i]-lb_local2)/(1e-8+abs(lb_local2)) <= reltol
                        bounds_touch2 = true
                    end
                end
        
                if bounds_touch2
                    disc_start[(var_count-1)*num_disc_points+j] = lb_local2
                    fix_disc_point[(var_count-1)*num_disc_points+j] = 1
                else
                    disc_start[(var_count-1)*num_disc_points+j] = point_vec2[i]
                end
                # println("point_vec2[",i,"]: ",point_vec2[i],"  fix_disc_point: ",(fix_disc_point[(var_count-1)*num_disc_points+j] > 0.5))
            end
        end

        m.best_bound = init_best_bound
        m.discretization = deepcopy(init_discretization)

        for j = 1:num_disc_vars
            tmp_vec = disc_start[(j-1)*num_disc_points+1:j*num_disc_points]
            tmp_vec2 = fix_disc_point[(j-1)*num_disc_points+1:j*num_disc_points]
            perm_sorted = sortperm(tmp_vec)
            disc_start[(j-1)*num_disc_points+1:j*num_disc_points] = tmp_vec[perm_sorted]
            fix_disc_point[(j-1)*num_disc_points+1:j*num_disc_points] = tmp_vec2[perm_sorted]
        end

        if Alp.get_option(m, :disc_sp_adaptive_part)
            count_fixed::Int = 0
            for j = 1:num_disc_vars*num_disc_points
                if fix_disc_point[j] > 0.5
                    count_fixed += 1
                    disc_upper_bound[j] = disc_lower_bound[j]
                end
            end
            println("\n  Fixed ",count_fixed," discretization points based on bounding heuristic\n")
        end
    else
        disc_specified_initial_guess = Alp.get_option(m,:disc_specified_initial_guess)
        curr_iter::Int = m.logs[:n_iter]+1
        for i = 1:num_disc_vars
            for j = 1:num_disc_points
                # disc_start[(i-1)*num_disc_points+j] = disc_lower_bound[(i-1)*num_disc_points+1] + (disc_upper_bound[(i-1)*num_disc_points+1] - disc_lower_bound[(i-1)*num_disc_points+1])*disc_specified_initial_guess[string(curr_iter)][(i-1)*num_disc_points+j]
                disc_start[(i-1)*num_disc_points+j] = disc_specified_initial_guess[string(curr_iter)][(i-1)*num_disc_points+j]

                bounds_touch = false    
                if Alp.get_option(m, :disc_sp_adaptive_part)
                    ub_local = disc_upper_bound[(i-1)*num_disc_points+j]
                    if disc_start[(i-1)*num_disc_points+j] >= ub_local || isapprox(disc_start[(i-1)*num_disc_points+j], ub_local; atol=abstol) || abs(disc_start[(i-1)*num_disc_points+j]-ub_local)/(1e-8+abs(ub_local)) <= reltol
                        bounds_touch = true
                    end
                    lb_local = disc_lower_bound[(i-1)*num_disc_points+j]
                    if disc_start[(i-1)*num_disc_points+j] <= lb_local || isapprox(disc_start[(i-1)*num_disc_points+j], lb_local; atol=abstol) || abs(disc_start[(i-1)*num_disc_points+j]-lb_local)/(1e-8+abs(lb_local)) <= reltol
                        bounds_touch = true
                    end

                    if bounds_touch
                        fix_disc_point[(i-1)*num_disc_points+j] = 1
                    end
                end
            end
        end

        if Alp.get_option(m, :disc_sp_adaptive_part)
            count_fixed = 0
            for j = 1:num_disc_vars*num_disc_points
                if fix_disc_point[j] > 0.5
                    count_fixed += 1
                    disc_upper_bound[j] = disc_lower_bound[j]
                end
            end
            println("\n  Fixed ",count_fixed," discretization points based on bounding heuristic and initial guess\n")
        end
    end

    return disc_start, disc_lower_bound, disc_upper_bound, fix_disc_point
end

"""
Function to check progress of MPBNGC for SP
"""
function mpbngc_sufficient_progress(objvals::Vector{Any}, epsvals::Vector{Any})
    diff_obj::Float64 = abs(objvals[end] - objvals[1])
    diff_eps::Float64 = sum(abs(epsvals[i+1] - epsvals[i]) for i=1:length(epsvals)-1)
    # println("diff_obj: ",diff_obj,"  diff_eps: ",diff_eps)
    # if diff_obj < 1E-04*abs(objvals[1]) || diff_eps < 1E-09
    if diff_eps < 1E-09
        return true
    end
    return false
end

"""
Method to compute strong partitioning points by solving the max_min problem
"""
function compute_sp_partition(m::Optimizer;kwargs...)

    start_time_sp = time()

    options = Dict(kwargs)

    haskey(options, :use_disc) ? discretization = options[:use_disc] : discretization = m.discretization
    haskey(options, :use_solution) ? point_vec = copy(options[:use_solution]) : point_vec = copy(m.best_bound_sol)

    if length(point_vec) < m.num_var_orig + m.num_var_linear_mip + m.num_var_nonlinear_mip
        point_vec = Alp.resolve_lifted_var_value(m, point_vec)
    end

	# store initial discretization and best lower bound
	init_discretization = deepcopy(discretization)
	init_best_bound = m.best_bound

    num_disc_vars::Int = length(m.disc_vars)
    num_disc_points::Int = Alp.get_option(m, :disc_sp_num_points)
    max_num_func_eval::Int = Alp.get_option(m, :disc_sp_budget)

    # Compute initial guess and bounds for partitioning points
    disc_lower_bound_orig, disc_upper_bound_orig = Alp.compute_sp_bounds(m,discretization,use_solution=point_vec)
    disc_start, disc_lower_bound, disc_upper_bound, fix_disc_point = Alp.compute_sp_initial_guess(m,discretization,point_vec)

    disc_min_spacing_factor = Alp.get_option(m, :disc_sp_min_spacing_factor)
    disc_min_spacing = zeros(Float64,num_disc_vars)

    # set bounds for the discretization points
    for i = 1:num_disc_vars
        disc_min_spacing[i] = (disc_upper_bound_orig[i] - disc_lower_bound_orig[i])/max(2*(num_disc_points+1),disc_min_spacing_factor)
        # println("var ",i,"  min spacing: ",disc_min_spacing[i])
        count_fixed_disc::Int = 0
        for j = 1:num_disc_points
            if Alp.get_option(m, :disc_sp_adaptive_part) && (fix_disc_point[(i-1)*num_disc_points+j] > 0.5)
                count_fixed_disc += 1
            else
                disc_lower_bound[(i-1)*num_disc_points+j] += (j-count_fixed_disc)*disc_min_spacing[i]
                disc_upper_bound[(i-1)*num_disc_points+j] -= (num_disc_points+1-j)*disc_min_spacing[i]
            end
        end
        # println("var i fix disc: ",fix_disc_point[(i-1)*num_disc_points+1:i*num_disc_points])
        # println("var i lbd: ",disc_lower_bound[(i-1)*num_disc_points+1:i*num_disc_points])
        # println("var i ubd: ",disc_upper_bound[(i-1)*num_disc_points+1:i*num_disc_points])
    end

    println("\nInitial guess for SP: ", disc_start,"\n")
    println("disc_lower_bound: ", disc_lower_bound,"\n")
    println("disc_upper_bound: ", disc_upper_bound,"\n")
    println("fix_disc_point: ", fix_disc_point,"\n\n")


    # Initialize best found discretization point and bound
    best_disc_bound::Float64 = -Inf
    curr_best_disc_bound::Float64 = -Inf
    if Alp.is_max_sense(m)
        best_disc_bound = Inf
        curr_best_disc_bound = Inf
    end
    best_disc_points = zeros(Float64,num_disc_points*num_disc_vars)

    num_func_eval::Int = 0

    if Alp.get_option(m, :disc_sp_method) == "mpbngc"

        num_curr_func_eval::Int = 0
        prev_objs = []
        prev_eps = []
        stalling_iter_check = 20
        stalling_restart = true
        bounding_error = false

        # specify the objective function for MPBNGC
        # function partitioning_objective_mpbngc(n, p, mm, f, g)
        function partitioning_objective_mpbngc(n, p, mm, f, g, eps_val, ierr)

            num_func_eval += 1
            num_curr_func_eval += 1
            processed = Set{Int}()

            for (var_count, i) in enumerate(m.disc_vars)
                if m.var_type[i] == :Cont
                    i in processed && continue
                    new_discretization = sort(vcat(init_discretization[i],p[(var_count-1)*num_disc_points+1:var_count*num_disc_points]))
                    discretization[i] = deepcopy(new_discretization)
                    push!(processed, i)
                else
                    error("Unexpected variable types while inserting partitions")
                end
            end

            mip_pointers = Alp.create_bounding_mip(m, use_disc=discretization, add_time=false)
            Alp.bounding_solve(m, add_time=false)
            mpbngc_obj::Float64 = m.best_bound

            if m.status[:bounding_solve] == MOI.OPTIMAL

                if Alp.is_max_sense(m)
                    f[1] = mpbngc_obj
                    if mpbngc_obj < best_disc_bound
                        best_disc_bound = mpbngc_obj
                        best_disc_points = deepcopy(p)
                    end
                    if mpbngc_obj < curr_best_disc_bound
                        curr_best_disc_bound = mpbngc_obj
                    end
                else
                    f[1] = -mpbngc_obj
                    if mpbngc_obj > best_disc_bound
                        best_disc_bound = mpbngc_obj
                        best_disc_points = deepcopy(p)
                    end
                    if mpbngc_obj > curr_best_disc_bound
                        curr_best_disc_bound = mpbngc_obj
                    end
                end

                # Check stalling
                append!(prev_objs, curr_best_disc_bound)
                append!(prev_eps, eps_val)
                if num_curr_func_eval > stalling_iter_check
                    prev_objs = prev_objs[2:end]
                    prev_eps = prev_eps[2:end]
                    stalling_restart = mpbngc_sufficient_progress(prev_objs, prev_eps)
                end

                if stalling_restart
                    println("Detected MPBNGC stalling for SP. Will restart MPBNGC with a new initial point")
                end

                # if num_func_eval%20 == 0
                #     println("disc: ", best_disc_points)
                # end

                α_opt = Dict()
                for k in keys(m.nonconvex_terms)
                    nl_type = m.nonconvex_terms[k][:nonlinear_type]
                    if (nl_type == :BILINEAR || nl_type == :MULTILINEAR)
                        for i in mip_pointers[:indices][k]
                            α_opt[i] = round.(JuMP.value.(mip_pointers[:part][i]))
                        end
                    elseif nl_type == :MONOMIAL
                        i = mip_pointers[:mon_indices][k]
                        α_opt[i] = round.(JuMP.value.(mip_pointers[:part][i]))
                    end
                end
                genl_grad, grad_status = Alp.bounding_solve_grad(m, mip_pointers, α_opt, discretization)
                if !(grad_status == MOI.OPTIMAL)
                    println("Ending SP solves because MIP solver returned the unexpected status ",grad_status," during bounding_solve_grad")
                    stalling_restart = true
                    bounding_error = true
                end
            else
                println("Ending SP solves because MIP solver returned the unexpected status ",m.status[:bounding_solve]," during bounding_solve")
                stalling_restart = true
                bounding_error = true
            end

            m.best_bound = init_best_bound

            # Check convergence
            bound_converged = false
            tol_shrink_factor = 0.01
            bound_cutoff = m.best_obj - tol_shrink_factor*Alp.get_option(m, :rel_gap)*(Alp.get_option(m, :tol) + abs(m.best_obj))
            if isapprox(m.best_obj, 0.0; atol = Alp.get_option(m, :tol))
                bound_cutoff = ((m.best_obj + 1) - tol_shrink_factor*Alp.get_option(m, :rel_gap)*(Alp.get_option(m, :tol) + (abs(m.best_obj) + 1))) - 1
            end
            if abs(m.best_obj) < Alp.get_option(m, :large_bound)
                if (Alp.is_max_sense(m) && best_disc_bound <= bound_cutoff) || (Alp.is_min_sense(m) && best_disc_bound >= bound_cutoff)
                    stalling_restart = false
                    bound_converged = true
                    ierr[1] = 0
                    println("Ending SP solves because bound converged with best_obj: ", m.best_obj,", bound: ", best_disc_bound)
                end
            end

            # Check time limit
            exceeded_time_limit = false
            if time() - start_time_sp > Alp.get_option(m, :disc_sp_time_limit)
                stalling_restart = false
                exceeded_time_limit = true
                println("Ending SP solves after running ",num_func_eval," iterations in ",round(time() - start_time_sp; digits=2)," seconds")
            end

            for (var_count, i) in enumerate(m.disc_vars)
                for j = 1:num_disc_points
                    if stalling_restart || exceeded_time_limit || bound_converged || bounding_error
                        ierr[1] = 8
                        g[(var_count-1)*num_disc_points+j,1] = 0.0
                    elseif Alp.is_max_sense(m)
                        g[(var_count-1)*num_disc_points+j,1] = genl_grad[i][j]
                    else
                        g[(var_count-1)*num_disc_points+j,1] = -genl_grad[i][j]
                    end
                end
            end

        end

        # Set up MPBNGC and solve the max-min problem
        C = zeros(Float64,num_disc_points*num_disc_vars,(num_disc_points-1)*num_disc_vars)
        for i = 1:num_disc_vars
            for j = 1:(num_disc_points-1)
                C[(i-1)*num_disc_points+j,(i-1)*(num_disc_points-1)+j] = 1.0
                C[(i-1)*num_disc_points+j+1,(i-1)*(num_disc_points-1)+j] = -1.0
            end
        end

        lbc = -Inf*ones(Float64,(num_disc_points-1)*num_disc_vars)
        ubc = zeros(Float64,(num_disc_points-1)*num_disc_vars)

        for i = 1:num_disc_vars
            for j = 1:(num_disc_points-1)
                if (!Alp.get_option(m, :disc_sp_adaptive_part)) || (fix_disc_point[(i-1)*num_disc_points+j+1] < 0.5)
                    ubc[(i-1)*(num_disc_points-1)+j] = -disc_min_spacing[i]
                end
            end
        end

        num_restarts::Int = -1
        max_restarts::Int = 100
        pert_frac::Float64 = 0.05
        Random.seed!(1234)  # fix random seed for reproducibility

        while stalling_restart && num_restarts < max_restarts && num_func_eval < max_num_func_eval

            num_restarts += 1
            # try one restart with the current best solution. otherwise, try restarts with random initializations
            if num_restarts > 1
                for i = 1:length(disc_start)
                    tmp_lower::Float64 = max(min((1-pert_frac)*disc_start[i] + pert_frac*disc_lower_bound[i], disc_upper_bound[i]), disc_lower_bound[i])
                    tmp_upper::Float64 = max(min((1-pert_frac)*disc_start[i] + pert_frac*disc_upper_bound[i], disc_upper_bound[i]), disc_lower_bound[i])
                    disc_start[i] = tmp_lower + rand()*(tmp_upper - tmp_lower)
                end
            end

            mpbngc_options = MPBNGCInterface.BundleOptions( OPT_LMAX	=> 20,
                                                            OPT_EPS		=> 1e-9,
                                                            OPT_FEAS 	=> 1e-9,
                                                            OPT_JMAX 	=> 5,
                                                            OPT_NITER	=> max_num_func_eval-num_func_eval,
                                                            OPT_NFASG	=> max_num_func_eval-num_func_eval,
                                                            OPT_NOUT	=> 6,
                                                            OPT_IPRINT	=> 3,
                                                            OPT_LOGLEVEL => 0,
                                                            OPT_GAM 	=> [0.5])

            prev_objs = []
            prev_eps = []
            stalling_restart = false
            num_curr_func_eval = 0
            curr_best_disc_bound = -Inf
            if Alp.is_max_sense(m)
                curr_best_disc_bound = Inf
            end

            prob = MPBNGCInterface.BundleProblem(num_disc_points*num_disc_vars, partitioning_objective_mpbngc, disc_start, disc_lower_bound, disc_upper_bound, lbc, ubc, C)
            MPBNGCInterface.solveProblem(prob, mpbngc_options)
            disc_start = deepcopy(best_disc_points)

            bounding_error = false

            println("Completed ", num_restarts," restarts.  stalling_restart = ", stalling_restart,".  num_func_eval = ", num_func_eval, "  best_disc_bound: ", best_disc_bound)
        end


    elseif Alp.get_option(m, :disc_sp_method) == "nlopt"

        # specify the objective function for NLopt
        function partitioning_objective_nlopt(x::Vector{Float64}, grad::Vector{Float64})

            num_func_eval += 1

            for (var_count, i) in enumerate(m.disc_vars)
                if m.var_type[i] == :Cont
                    new_discretization = sort(vcat(init_discretization[i],x[(var_count-1)*num_disc_points+1:var_count*num_disc_points]))
                    discretization[i] = deepcopy(new_discretization)
                else
                    error("Unexpected variable types while inserting partitions")
                end
            end

            Alp.create_bounding_mip(m, use_disc=discretization, add_time=false)
            Alp.bounding_solve(m, add_time=false)
            nlopt_obj::Float64 = m.best_bound

            if Alp.is_max_sense(m)
                if nlopt_obj < best_disc_bound
                    best_disc_bound = nlopt_obj
                    best_disc_points = deepcopy(x)
                end
            else
                if nlopt_obj > best_disc_bound
                    best_disc_bound = nlopt_obj
                    best_disc_points = deepcopy(x)
                end
            end

            m.best_bound = init_best_bound

            if num_func_eval % 20 == 1
                println("NLopt iter#",num_func_eval,"  best obj: ",round(best_disc_bound; digits=6),"  for x = ",round.(best_disc_points; digits=4))
            end

            # Check convergence
            bound_converged::Bool = false
            tol_shrink_factor::Float64 = 0.99
            if abs(m.best_obj) < Alp.get_option(m, :large_bound)
                if (Alp.is_max_sense(m) && best_disc_bound <= m.best_obj) || (Alp.is_min_sense(m) && best_disc_bound >= m.best_obj) || (abs(m.best_obj - best_disc_bound) <= tol_shrink_factor*Alp.get_option(m, :rel_gap)*(Alp.get_option(m, :tol) + abs(m.best_obj)))
                    bound_converged = true
                    println("Ending SP solves because bound converged with best_obj: ", m.best_obj,", bound: ", best_disc_bound)
                end
            end

            # Check time limit
            exceeded_time_limit::Bool = false
            if time() - start_time_sp > Alp.get_option(m, :disc_sp_time_limit)
                exceeded_time_limit = true
                println("Ending SP solves after running ",num_func_eval," iterations in ",round(time() - start_time_sp; digits=2)," seconds")
                return
            end

            # Check time limit
            if bound_converged || exceeded_time_limit
                for (var_count, i) in enumerate(m.disc_vars)
                    for j = 1:num_disc_points
                        grad[(var_count-1)*num_disc_points+j] = 0.0
                    end
                end    
            end

            return nlopt_obj
        end

        # Set up NLopt and solve the max-min problem
        # nlopt_opt = NLopt.Opt(:LN_BOBYQA, num_disc_points*num_disc_vars)
        nlopt_opt = NLopt.Opt(:LN_NEWUOA_BOUND, num_disc_points*num_disc_vars)
        if Alp.is_min_sense(m)
            nlopt_opt.max_objective = partitioning_objective_nlopt
        else
            nlopt_opt.min_objective = partitioning_objective_nlopt
        end
        nlopt_opt.lower_bounds = disc_lower_bound
        nlopt_opt.upper_bounds = disc_upper_bound
        nlopt_opt.maxeval = max_num_func_eval

        NLopt.optimize!(nlopt_opt,disc_start)

    else
        error("Unknown method for strong partitioning")
    end

    println("\nBest found disc_points using ", Alp.get_option(m, :disc_sp_method), " (without postprocessing): ",round.(best_disc_points; digits=16),"  with bound: ",round(best_disc_bound; digits=16),"  in time: ",round(time()-start_time_sp; digits=2),"s")


    # Postprocessing to try and eliminate partitioning points that are not useful
    if Alp.get_option(m, :disc_sp_postprocessing)
        start_time_sp_post = time()
        println("\n  Starting postprocessing to try and eliminate partitioning points that are not useful...")
        count_excluded::Int = 0
        tmp_best_disc_points = deepcopy(best_disc_points)

        processed = Set{Int}()
        tmp_discretization = deepcopy(init_discretization)
        for (var_count, i) in enumerate(m.disc_vars)
            i in processed && continue
            new_discretization = sort(vcat(init_discretization[i],best_disc_points[(var_count-1)*num_disc_points+1:var_count*num_disc_points]))
            tmp_discretization[i] = deepcopy(new_discretization)
            push!(processed, i)
        end

        for (var_count, i) in enumerate(m.disc_vars)
            if time() - start_time_sp > Alp.get_option(m, :disc_sp_time_limit)
                break
            end

            excluded_points = zeros(Int64,num_disc_points)

            for j = 1:num_disc_points
                excluded_points[j] = 1
                count_excluded += 1

                tmp_discretization_2 = deepcopy(tmp_discretization)
                tmp_disc_points = []
                for k = 1:num_disc_points
                    append!(tmp_disc_points,init_discretization[i][1])
                    if excluded_points[k] < 0.5
                        append!(tmp_disc_points,best_disc_points[(var_count-1)*num_disc_points+k])
                    end
                    append!(tmp_disc_points,init_discretization[i][2])
                end
                tmp_discretization_2[i] = convert(Vector{Float64}, sort(tmp_disc_points))
                # tmp_discretization_2[i] = sort(tmp_disc_points)

                if Alp.get_option(m,:disc_sp_adaptive_part) && (fix_disc_point[(var_count-1)*num_disc_points+j] > 0.5)
                    curr_bound = best_disc_bound
                else
                    Alp.create_bounding_mip(m, use_disc=tmp_discretization_2, add_time=false)
                    Alp.bounding_solve(m, add_time=false)
                    curr_bound = m.best_bound
                    m.best_bound = init_best_bound
                end
                excluded_points[j] = 0
                count_excluded -= 1

                tol_shrink_factor = 0.01
                if abs(curr_bound-best_disc_bound) < tol_shrink_factor*Alp.get_option(m, :rel_gap)*abs(best_disc_bound)
                    tmp_best_disc_points[(var_count-1)*num_disc_points+j] = init_discretization[i][1]
                    count_excluded += 1
                    excluded_points[j] = 1
                    tmp_discretization = deepcopy(tmp_discretization_2)
                end
            end
            tmp_vec = sort!(tmp_best_disc_points[(var_count-1)*num_disc_points+1:var_count*num_disc_points])
            tmp_best_disc_points[(var_count-1)*num_disc_points+1:var_count*num_disc_points] = tmp_vec
        end

        best_disc_points = deepcopy(tmp_best_disc_points)


        println("Time for postprocessing: ",round(time()-start_time_sp_post; digits=2),"s\n")
        println("\nBest found disc_points using ", Alp.get_option(m, :disc_sp_method), " (with postprocessing): ",round.(best_disc_points; digits=16),"  with bound: ",round(best_disc_bound; digits=16),"  in time: ",round(time()-start_time_sp; digits=2),"s. Excluded ",count_excluded," discretization points")


        if Alp.get_option(m, :disc_sp_postprocessing_pert)
            start_time_sp_post_pert = time()
            println("")
            pert_frac_postprocess = 0.2
            pert_step_postprocess = pert_frac_postprocess/20
            pert_length = 21
            abstol = Alp.get_option(m, :disc_abs_width_tol)
            for i = 1:num_disc_vars
                for j = 1:num_disc_points
                    tmp_lower = max(disc_lower_bound[(i-1)*num_disc_points+j], best_disc_points[(i-1)*num_disc_points+j] - pert_frac_postprocess*(disc_upper_bound_orig[i] - disc_lower_bound_orig[i]))
                    tmp_upper = min(disc_upper_bound[(i-1)*num_disc_points+j], best_disc_points[(i-1)*num_disc_points+j] + pert_frac_postprocess*(disc_upper_bound_orig[i] - disc_lower_bound_orig[i]))

                    list_points_1 = [range(tmp_lower, best_disc_points[(i-1)*num_disc_points+j], length=pert_length);]
                    list_points_2 = [range(best_disc_points[(i-1)*num_disc_points+j], tmp_upper, length=pert_length);]
                    list_points = [list_points_1; list_points_2[2:end]]
                    list_bounds = zeros(Float64,length(list_points))
 
                    if (!isapprox(best_disc_points[(i-1)*num_disc_points+j], disc_lower_bound_orig[i]; atol=abstol)) && (!isapprox(best_disc_points[(i-1)*num_disc_points+j], disc_upper_bound_orig[i]; atol=abstol))
                        for (k, pt) in enumerate(list_points)
                            tmp_best_disc_points_pp = deepcopy(best_disc_points)
                            tmp_best_disc_points_pp[(i-1)*num_disc_points+j] = pt

                            processed_pp = Set{Int}()
                            tmp_discretization_pp = deepcopy(init_discretization)
                            for (var_count, l) in enumerate(m.disc_vars)
                                l in processed_pp && continue
                                new_discretization_pp = sort(vcat(init_discretization[l],tmp_best_disc_points_pp[(var_count-1)*num_disc_points+1:var_count*num_disc_points]))
                                tmp_discretization_pp[l] = deepcopy(new_discretization_pp)
                                push!(processed_pp, l)
                            end

                            Alp.create_bounding_mip(m, use_disc=tmp_discretization_pp, add_time=false)
                            Alp.bounding_solve(m, add_time=false)
                            list_bounds[k] = m.best_bound
                            m.best_bound = init_best_bound
                        end

                        println("Postprocessing Sensitivity for discretization point ",(i-1)*num_disc_points+j," :  points: ",round.(list_points;digits=16),"  :  bounds: ",round.(list_bounds;digits=16),"\n")
                        end
                end
            end
            println("")
            println("Time for postprocessing perturbation analysis: ",round(time()-start_time_sp_post_pert; digits=2),"s\n")
        end

    end

    m.discretization = deepcopy(init_discretization)

    return best_disc_points
end

"""
Method to add user-specified partitions
"""
function add_specified_partitions(m::Optimizer,discretization::Dict,part_points::Vector{Float64})

    num_disc_points::Int64 = round(length(part_points)/length(m.disc_vars))
    if num_disc_points*length(m.disc_vars) != length(part_points)
        error("The dimension of user-specified partition points is unexpected!")
    end

    # point_vec = Dict()
    # for i in m.disc_vars
    #     point_vec[i] = 0.5*(discretization[i][1] + discretization[i][2])
    # end

    # disc_lower_bound, disc_upper_bound = Alp.compute_sp_bounds(m,discretization, use_solution=point_vec)

    # println("m.disc_vars: ", m.disc_vars)

    println("Added the following user-specified partitions:")
    for (var_count, i) in enumerate(m.disc_vars)
        for k = 1:num_disc_points
            λCnt = length(discretization[i])
            # point = disc_lower_bound[var_count] + (disc_upper_bound[var_count] - disc_lower_bound[var_count] + 1E-12)*part_frac[(var_count-1)*num_disc_points+k]
            point = part_points[(var_count-1)*num_disc_points+k]

            if m.var_type[i] == :Cont
                point = correct_point(m, discretization[i], point, i)
                for j in 1:λCnt
                    if point >= discretization[i][j] && point <= discretization[i][j+1]
                        insert_single_partition(m, j, point, discretization[i])
                        break
                    end
                end
            else
                error("Unexpected variable types while inserting partitions")
            end
        end
        println("var ",i,": ",round.(discretization[i];digits=6))
    end

    return discretization
end
