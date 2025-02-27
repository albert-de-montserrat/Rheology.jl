function eval_residual(x, composite_expanded, composite_global, state_funs, unique_funs_global, subtractor_vars, inds_x_to_subtractor, inds_args_to_x, args_template, args_all, args_solve, args_other)
    subtractor        = generate_subtractor(x, inds_x_to_subtractor)
    args_tmp          = generate_args_from_x(x, inds_args_to_x, args_template, args_all)
    args_solve2       = merge(update_args2(args_solve, x), args_other)
    subtractor_global = generate_subtractor_global(composite_global, state_funs, unique_funs_global, args_solve2)
    
    eval_state_functions(state_funs, composite_expanded, args_tmp) - subtractor - subtractor_vars + subtractor_global
end

@generated function generate_subtractor(x::SVector{N, T}, inds::NTuple{N, Int}) where {N,T}
    quote
        @inline 
        Base.@nexprs $N i -> x_i = begin
            ind = @inbounds inds[i]
            iszero(ind) ? zero(T) : @inbounds(x[ind])
        end
        Base.@ncall $N SVector x
    end 
end

@inline generate_subtractor(x::SVector{N, T}, ::Tuple{}) where {N, T} = zero(x)

@generated function generate_args_from_x(x::SVector{N}, inds_args_to_x, args_template, args_all) where N
    quote
        @inline 
        Base.@ntuple $N i -> begin
            ind = @inbounds inds_args_to_x[i]
            xᵢ  = getindex_tuple(x, ind) 
            _generate_args_from_x(xᵢ, @inbounds(args_template[i]), @inbounds(args_all[i]))
        end
    end
end
@inline generate_args_from_x(::SVector, ::Tuple{}, ::Any, ::Tuple{}) = (;)
@inline generate_args_from_x(::SVector, ::Any, ::Any, ::Tuple{}) = (;)

@inline _generate_args_from_x(xᵢ::NTuple{N, Number}, ::Tuple{}, ::Any) where N = xᵢ

function _generate_args_from_x(xᵢ::NTuple{N, Number}, args_templateᵢ::NamedTuple, args_allᵢ::NamedTuple) where N
    tmp = (; zip(keys(args_templateᵢ), xᵢ)...)
    merge(args_allᵢ, tmp)
end


@generated function generate_subtractor_global(composite_global::NTuple{N3, AbstractRheology}, state_funs::NTuple{N1, Any}, unique_funs_global::NTuple{N2, Any}, args_solve) where {N1, N2, N3}
    quote
        @inline
        type = eltype(args_solve)
        v = Base.@ntuple $N1 i -> begin
            val = zero(type)
            if state_funs[i] === state_var_reduction
                fn = unique_funs_global[i]
                Base.@nexprs $N3 k -> begin
                    val += fn(composite_global[k], args_solve)
                end
            end
            val
        end
        SVector{$N1, type}(v)
    end
end

@inline generate_subtractor_global(::Tuple{}, ::NTuple{N1, Any}, ::NTuple{N2, Any}, ::Any) where {N1, N2} = @SVector zeros(N1)
@inline generate_subtractor_global(::NTuple{N1, AbstractRheology}, ::Tuple{}, ::NTuple{N2, Any}, ::Any) where {N1, N2} = @SVector zeros(N2)
