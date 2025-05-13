# ----- Variables ----- #
@variable(m2, objvar)
x_Idx = Any[1]
@variable(m2, x[x_Idx])
set_lower_bound(x[1], 0.0)
set_upper_bound(x[1], 2.0)


# ----- Constraints ----- #
@constraint(m2, e1, 2*x[1]+objvar == 0.0)
@constraint(m2, e2, x[1] <= 1.0)


# ----- Objective ----- #
@objective(m2, Min, objvar)
