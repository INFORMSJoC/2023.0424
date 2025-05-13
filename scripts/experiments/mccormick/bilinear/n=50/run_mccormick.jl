using JuMP 
using Gurobi


const gurobi = optimizer_with_attributes(Gurobi.Optimizer,
					MOI.Silent() => true,
					"MIPGap" => 1E-06,
					"Presolve" => -1)

global m2 = Model(gurobi)
include("../../warmup_lp.jl")
JuMP.optimize!(m2)


inst = ARGS[1]
modelFile = "../../../../../data/bilinear/n=50/instances/mccormick/qcqp_v50_b250_s100_$(inst)_mccormick.jl"


global m = Model(gurobi)
include(modelFile)


JuMP.optimize!(m)
println("x_opt: ", JuMP.value.(x))
println("w_opt: ", JuMP.value.(w))
println("solution time: ", JuMP.solve_time(m))
println("McCormick lower bound: ", JuMP.objective_value(m),"\n")
