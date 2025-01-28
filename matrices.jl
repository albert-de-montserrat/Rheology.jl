@generated function compute_residual(composite::NTuple{N, Any}, statefuns, vars, args) where N
    quote
        R = -SA[values(vars)...]
        Base.@nexprs $N i -> begin # this will be put into a function, hardcoded for now
            R += eval_state_functions(statefuns, composite[i], args)
        end
        R
    end
end

@generated function compute_jacobian(x, composite::NTuple{N1, Any}, statefuns::NTuple{N2, Any}, args_diff, args_nondiff) where {N1,N2}
    quote
        J = @SMatrix zeros(N2, N2)
        Base.@nexprs $N1 i -> begin # this will be put into a function, hardcoded for now
            J += _compute_jacobian(x, composite[i], statefuns,  args_diff, args_nondiff)
        end
        J
    end
end

function _compute_jacobian(x::SVector{N}, compositeᵢ, statefuns, args_diff::NamedTuple{na}, args_nondiff::NamedTuple{nb}) where {N, na, nb}
    f = x -> begin
        k = keys(args_diff)
        v = tuple(x...)
        args_diff2 = (; zip(k, v)...)
        args = merge(args_diff2, args_nondiff)
        eval_state_functions(statefuns, compositeᵢ, args)    
    end
    return ForwardDiff.jacobian(x -> f(x), x)
end

