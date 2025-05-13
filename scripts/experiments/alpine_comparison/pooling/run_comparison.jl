# Base pooling problem model

using JuMP, Gurobi, KNITRO, Alpine, SparseArrays, JSON


gurobi = optimizer_with_attributes(Gurobi.Optimizer, MOI.Silent() => true, "MIPGap" => 1E-06, "Presolve" => -1)
knitro = optimizer_with_attributes(KNITRO.Optimizer, "algorithm" => 3, MOI.Silent() => true)



alpine_orig = optimizer_with_attributes(Alpine.Optimizer,
                                        "nlp_solver" => knitro,
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

modelFile = "../../../../data/pooling/pooling_instances/pooling/random_schweiger_c15_e150_q1_$(inst).jl"
partFile_mpbngc = "../../../../results/pooling/strong_partitioning_points/random_schweiger_c15_e150_q1_$(inst)_part_mpbngc.json"
partFile_ml = "../../../../results/pooling/ml_pred/random_schweiger_c15_e150_q1_$(inst)_ab_pred.json"


dict_features_mpbngc = JSON.parse(join(readlines(partFile_mpbngc)))
dict_features_ml = JSON.parse(join(readlines(partFile_ml)))

part_point_mpbngc = convert(Vector{Float64},dict_features_mpbngc["part_mpbngc"])
part_point_ml = convert(Vector{Float64},dict_features_ml["part_points"])

part_dict_mpbngc = Dict("1" => part_point_mpbngc)
part_dict_ml = Dict("1" => part_point_ml)






include(modelFile)



mod = Model(alpine_orig)


@variables(mod, begin
	0 <= w[(i,l,j) in arcs_ilj] <= min(cap_i[i],cap_l[l],cap_j[j])
	0 <= x_il[(i,l) in arcs_il] <= min(cap_i[i],cap_l[l])
	0 <= x_lj[(l,j) in arcs_lj] <= min(cap_l[l],cap_j[j])
	0 <= x_ij[(i,j) in arcs_ij] <= min(cap_i[i],cap_j[j])
	0 <= q[arcs_il] <= 1
end)


@constraints(mod, begin
	capi[i=1:num_i], sum(x_ij[(i,j)] for j in outputs_from_input[i]) + sum(x_il[(i,l)] for l in pools_from_input[i]) <= cap_i[i]

	capl[l=1:num_l], sum(x_lj[(l,j)] for j in outputs_from_pool[l]) <= cap_l[l]

	capj[j=1:num_j], sum(x_ij[(i,j)] for i in inputs_to_output[j]) + sum(x_lj[(l,j)] for l in pools_to_output[j]) <= cap_j[j]

	sumfrac[l=1:num_l], sum(q[(i,l)] for i in inputs_to_pool[l]) == 1

	eq1[(i,l) in arcs_il], x_il[(i,l)] == sum(w[(i,l,j)] for j in outputs_from_pool[l])

        specdown[k=1:num_k, j=1:num_j], sum(gamma_low[i][j][k]*w[(i,l,j)] for (i,l) in inputs_pools_to_output[j]) + sum(gamma_low[i][j][k]*x_ij[(i,j)] for i in inputs_to_output[j]) >= 0
	
	specup[k=1:num_k, j=1:num_j], sum(gamma_up[i][j][k]*w[(i,l,j)] for (i,l) in inputs_pools_to_output[j]) + sum(gamma_up[i][j][k]*x_ij[(i,j)] for i in inputs_to_output[j]) <= 0

	rlt1[(l,j) in arcs_lj], sum(w[(i,l,j)] for i in inputs_to_pool[l]) == x_lj[(l,j)]

	rlt2[(i,l) in arcs_il], sum(w[(i,l,j)] for j in outputs_from_pool[l]) <= cap_l[l]*q[(i,l)]
end)


@NLconstraint(mod, blin[(i,l,j) in arcs_ilj], w[(i,l,j)] == q[(i,l)]*x_lj[(l,j)])


@objective(mod, Min, sum(cost_ij[i,j]*x_ij[(i,j)] for (i,j) in arcs_ij) + sum(cost_il[i,l]*x_il[(i,l)] for (i,l) in arcs_il) + sum(cost_lj[l,j]*x_lj[(l,j)] for (l,j) in arcs_lj) )



# THIRD RUN: DEFAULT ALPINE

JuMP.optimize!(mod)



println("objective value: ", JuMP.objective_value(mod))
println("objective bound: ", JuMP.objective_bound(mod))
println("solution time: ", JuMP.solve_time(mod))
println("w_opt: ", JuMP.value.(w))
println("x_ij_opt: ", JuMP.value.(x_ij))
println("x_il_opt: ", JuMP.value.(x_il))
println("x_lj_opt: ", JuMP.value.(x_lj))
println("q_opt: ", JuMP.value.(q))
println("\n\n")
sleep(30)







alpine_sp = optimizer_with_attributes(Alpine.Optimizer,
                                        "nlp_solver" => knitro,
                                        "mip_solver" => gurobi,
                                        "presolve_bt" => false,
                                        "disc_var_pick" => 0,
                                        "disc_partition_at_existing" => true,
                                        "disc_sp_iter" => 1,
                                        "disc_sp_num_points" => 2,
                                        "disc_use_specified_part" => true,
                                        "disc_specified_part" => part_dict_mpbngc,
                                        "time_limit" => 7200)


# FOURTH RUN: ALPINE+SP

JuMP.set_optimizer(mod, alpine_sp)
JuMP.optimize!(mod)


println("objective value: ", JuMP.objective_value(mod))
println("objective bound: ", JuMP.objective_bound(mod))
println("solution time: ", JuMP.solve_time(mod))
println("w_opt: ", JuMP.value.(w))
println("x_ij_opt: ", JuMP.value.(x_ij))
println("x_il_opt: ", JuMP.value.(x_il))
println("x_lj_opt: ", JuMP.value.(x_lj))
println("q_opt: ", JuMP.value.(q))
println("\n\n")
sleep(30)







alpine_ml = optimizer_with_attributes(Alpine.Optimizer,
                                        "nlp_solver" => knitro,
                                        "mip_solver" => gurobi,
                                        "presolve_bt" => false,
                                        "disc_var_pick" => 0,
                                        "disc_partition_at_existing" => true,
                                        "disc_sp_iter" => 1,
                                        "disc_sp_num_points" => 2,
                                        "disc_use_specified_part" => true,
                                        "disc_specified_part" => part_dict_ml,
                                        "time_limit" => 7200)


# FIFTH RUN: ALPINE+ML

JuMP.set_optimizer(mod, alpine_ml)
JuMP.optimize!(mod)


println("objective value: ", JuMP.objective_value(mod))
println("objective bound: ", JuMP.objective_bound(mod))
println("solution time: ", JuMP.solve_time(mod))
println("w_opt: ", JuMP.value.(w))
println("x_ij_opt: ", JuMP.value.(x_ij))
println("x_il_opt: ", JuMP.value.(x_il))
println("x_lj_opt: ", JuMP.value.(x_lj))
println("q_opt: ", JuMP.value.(q))
println("\n\n")

