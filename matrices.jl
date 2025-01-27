@generated function compute_residual(composite::NTuple{N, Any}, statefuns, vars, args) where N
    quote
        R = -SA[values(vars)...]
        Base.@nexprs $N i -> begin # this will be put into a function, hardcoded for now
            R += eval_state_functions(statefuns, composite[i], args)
        end
        R
    end
end

@generated function compute_jacobian(x, composite::NTuple{N1, Any}, statefuns::NTuple{N2, Any}, args) where {N1,N2}
    quote
        J = @SMatrix zeros(N2, N2)
        Base.@nexprs $N1 i -> begin # this will be put into a function, hardcoded for now
            J += _compute_jacobian(x, composite[i], statefuns, args)
        end
        J
    end
end

function _compute_jacobian(x::SVector{N}, compositeᵢ, statefuns, args) where N
    f = x -> begin
        k = keys(args)
        v = values(args)
        nondiff_vals = v[N+1:end]
        diff_vals = tuple(x..., nondiff_vals...)
        diff_args = (; zip(k, diff_vals)...)
        eval_state_functions(statefuns, compositeᵢ, diff_args)    
    end
    return ForwardDiff.jacobian(x -> f(x), x)
end

# compute_jacobian(x, composite, statefuns, args)
# @b compute_jacobian($(x, composite, statefuns, args)...)

# # compute the global jacobian and residual
# R = -SA[values(vars)...]
# J = @SMatrix zeros(length(statefuns), length(statefuns))
# Base.@nexprs 3 i -> begin # this will be put into a function, hardcoded for now
#     J += ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[i], (; τ = x[1], P = x[2], dt = dt)), x)
#     R += eval_state_functions(statefuns, composite[i], args)
# end
