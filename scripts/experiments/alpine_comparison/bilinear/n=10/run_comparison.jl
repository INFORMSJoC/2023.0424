using Alpine 
using JuMP 
using Gurobi
using Ipopt
using JSON



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
                                         "disc_partition_at_existing" => true,
                                         "time_limit" => 7200)




# FIRST WARMUP RUN

global m = Model(alpine_orig)
include("../../warmup_qcqp.jl")
JuMP.optimize!(m)

println("\n\n")
sleep(30)




# SECOND WARMUP RUN

warmup_part_point_mpbngc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1912521764462802, 0.5]
warmup_part_dict_mpbngc = Dict("1" => warmup_part_point_mpbngc)

warmup_alpine_sp_mpbngc = optimizer_with_attributes(Alpine.Optimizer,
                                         "nlp_solver" => ipopt,
                                         "mip_solver" => gurobi,
                                         "presolve_bt" => false,
                                         "disc_var_pick" => 0,
                                         "disc_partition_at_existing" => true,
                                         "disc_sp_iter" => 1,
                                         "disc_sp_num_points" => 2,
                                         "disc_use_specified_part" => true,
                                         "disc_specified_part" => warmup_part_dict_mpbngc,
                                         "time_limit" => 7200)


global m = nothing
global m = Model(warmup_alpine_sp_mpbngc)
include("../../warmup_qcqp.jl")


JuMP.optimize!(m)
println("\n\n")
sleep(30)




inst = ARGS[1]

modelFile = "../../../../../data/bilinear/n=10/instances/bilinear/qcqp_v10_b45_s100_$(inst).jl"
partFile_mpbngc = "../../../../../results/bilinear/n=10/strong_partitioning_points/qcqp_v10_b45_s100_$(inst)_part_mpbngc.json"
partFile_ml = "../../../../../results/bilinear/n=10/ml_pred/qcqp_v10_b45_s100_$(inst)_ab_pred.json"


dict_features_mpbngc = JSON.parse(join(readlines(partFile_mpbngc)))
dict_features_ml = JSON.parse(join(readlines(partFile_ml)))

part_point_mpbngc = convert(Vector{Float64},dict_features_mpbngc["part_mpbngc"])
part_point_ml = convert(Vector{Float64},dict_features_ml["part_points"])

part_dict_mpbngc = Dict("1" => part_point_mpbngc)
part_dict_ml = Dict("1" => part_point_ml)



alpine_sp_mpbngc = optimizer_with_attributes(Alpine.Optimizer,
                                         "nlp_solver" => ipopt,
                                         "mip_solver" => gurobi,
                                         "presolve_bt" => false,
                                         "disc_var_pick" => 0,
                                         "disc_partition_at_existing" => true,
                                         "disc_sp_iter" => 1,
                                         "disc_sp_num_points" => 2,
                                         "disc_use_specified_part" => true,
                                         "disc_specified_part" => part_dict_mpbngc,
                                         "time_limit" => 7200)


alpine_sp_ml = optimizer_with_attributes(Alpine.Optimizer,
                                         "nlp_solver" => ipopt,
                                         "mip_solver" => gurobi,
                                         "presolve_bt" => false,
                                         "disc_var_pick" => 0,
                                         "disc_partition_at_existing" => true,
                                         "disc_sp_iter" => 1,
                                         "disc_sp_num_points" => 2,
					                     "disc_use_specified_part" => true,
					                     "disc_specified_part" => part_dict_ml,
                                         "time_limit" => 7200)




# THIRD RUN: DEFAULT ALPINE

global m = nothing
global m = Model(alpine_orig)
include(modelFile)


JuMP.optimize!(m)
println("objective value: ", JuMP.objective_value(m))
println("objective bound: ", JuMP.objective_bound(m))
println("x_opt: ", JuMP.value.(x))
println("solution time: ", JuMP.solve_time(m),"\n\n\n")

sleep(30)




# FOURTH RUN: ALPINE+SP

global m = nothing
global m = Model(alpine_sp_mpbngc)
include(modelFile)


JuMP.optimize!(m)
println("objective value: ", JuMP.objective_value(m))
println("objective bound: ", JuMP.objective_bound(m))
println("x_opt: ", JuMP.value.(x))
println("solution time: ", JuMP.solve_time(m),"\n\n\n")

sleep(30)




# FIFTH RUN: ALPINE+ML

global m = nothing
global m = Model(alpine_sp_ml)
include(modelFile)


JuMP.optimize!(m)
println("objective value: ", JuMP.objective_value(m))
println("objective bound: ", JuMP.objective_bound(m))
println("x_opt: ", JuMP.value.(x))
println("solution time: ", JuMP.solve_time(m),"\n")

