# Base pooling problem model

using JuMP, Gurobi, SparseArrays


gurobi = optimizer_with_attributes(Gurobi.Optimizer, MOI.Silent() => true, "MIPGap" => 1E-06, "Presolve" => -1)



global m2 = Model(gurobi)
include("../warmup_lp.jl")
JuMP.optimize!(m2)
sleep(20)



inst = ARGS[1]
modelFile = "../../../../data/pooling/pooling_instances/pooling/random_schweiger_c15_e150_q1_$(inst).jl"

include(modelFile)



mod = Model(gurobi)


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


@constraint(mod, mcc1[(i,l,j) in arcs_ilj], w[(i,l,j)] >= q[(i,l)]*min(cap_l[l],cap_j[j]) + x_lj[(l,j)] - min(cap_l[l],cap_j[j]))
@constraint(mod, mcc2[(i,l,j) in arcs_ilj], w[(i,l,j)] <= x_lj[(l,j)])
@constraint(mod, mcc3[(i,l,j) in arcs_ilj], w[(i,l,j)] <= q[(i,l)]*min(cap_l[l],cap_j[j]))


@objective(mod, Min, sum(cost_ij[i,j]*x_ij[(i,j)] for (i,j) in arcs_ij) + sum(cost_il[i,l]*x_il[(i,l)] for (i,l) in arcs_il) + sum(cost_lj[l,j]*x_lj[(l,j)] for (l,j) in arcs_lj) )


JuMP.optimize!(mod)



println("McCormick lower bound: ", JuMP.objective_value(mod),"\n")
println("solution time: ", JuMP.solve_time(mod),"\n")
println("w_opt: ", JuMP.value.(w))
println("x_ij_opt: ", JuMP.value.(x_ij))
println("x_il_opt: ", JuMP.value.(x_il))
println("x_lj_opt: ", JuMP.value.(x_lj))
println("q_opt: ", JuMP.value.(q))
println("\n\n")
