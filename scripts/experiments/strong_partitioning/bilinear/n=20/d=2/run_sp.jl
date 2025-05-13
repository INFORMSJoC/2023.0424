using Alpine 
using JuMP 
using Gurobi
using Ipopt


const gurobi = optimizer_with_attributes(Gurobi.Optimizer,
					MOI.Silent() => true,
					"MIPGap" => 1E-06,
					"Presolve" => -1)

const ipopt = optimizer_with_attributes(Ipopt.Optimizer, 
                                        MOI.Silent() => true, 
                                        "sb" => "yes", 
                                        "max_iter" => 10000)

const alpine_orig = optimizer_with_attributes(Alpine.Optimizer,
                                         "nlp_solver" => ipopt,
                                         "mip_solver" => gurobi,
                                         "presolve_bt" => false,
                                         "disc_var_pick" => 0,
                                         "time_limit" => 7200)

const alpine_sp = optimizer_with_attributes(Alpine.Optimizer,
                                         "nlp_solver" => ipopt,
                                         "mip_solver" => gurobi,
                                         "presolve_bt" => false,
                                         "disc_var_pick" => 0,
                                         "disc_sp_iter" => 1,
                                         "disc_sp_num_points" => 2,
                                         "max_iter" => 1,
                                         "disc_sp_min_spacing_factor" => Inf,
                                         "disc_sp_adaptive_part" => true,
                                         "disc_sp_postprocessing" => true, 
                                         "disc_sp_budget" => 500,
                                         "time_limit" => 7200)



global m = Model(alpine_orig)
include("warmup_qcqp.jl")
JuMP.optimize!(m)

println("\n\n")

sleep(30)


inst = ARGS[1]
modelFile = "../../../../../../data/bilinear/n=20/instances/bilinear/qcqp_v20_b100_s100_$(inst).jl"



global m = nothing
global m = Model(alpine_sp)
include(modelFile)


JuMP.optimize!(m)
