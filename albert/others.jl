@inline state_var_reduction(::AbstractRheology, x::NTuple{N, T}) where {T<:Number, N} = sum(x[i] for i in 1:N)

@inline merge_funs(funs1::NTuple{N1, Any}, funs2::NTuple{N2, Any}) where {N1, N2} = (funs1..., funs2...)

@generated function reduction_funs_args_indices(funs_local::NTuple{N1, Any}, unique_funs_local::NTuple{N2, Any}) where {N1, N2}
    quote
        @inline 
        Base.@ntuple $N2 i ->  begin
            ind = Base.@ntuple $N1 j-> begin
                unique_funs_local[i] == funs_local[j] ? j : ()
            end
            Base.IteratorsMD.flatten(ind)
        end
    end
end

function generate_indices_from_args_to_x(::NTuple{N, Any}, reduction_ind, ::Val{N_reductions}) where {N, N_reductions}
    inds_global = ntuple(Val(N_reductions)) do i 
        @inline
         i ≤ N ? reduction_ind[i] .+ N_reductions : (0,)
    end
    inds_local = ntuple(x -> (x + N_reductions,), Val(N))
    
    tuple(inds_global...,  inds_local...)
end

@generated function mapping_x_to_subtractor(state_funs::NTuple{N1, Any}, unique_funs_local::NTuple{N2, Any}) where {N1, N2}
    quote
        @inline 
        Base.@ntuple $N1 i -> begin
            ind = 0
            Base.@nexprs $N2 j -> begin
                @inbounds check = unique_funs_local[j] === state_funs[i]
                check && (ind = j)
            end
            ind
        end
    end
end

@inline function expand_series_composite(composite::NTuple{N, AbstractRheology}, funs_local, ::Val{N_reductions}) where {N, N_reductions}
    (ntuple(_-> first(composite), Val(N_reductions))..., expand_composite(composite, funs_local, series_state_functions)...)
end

@inline function expand_series_composite(composite::NTuple{N, AbstractRheology}, ::Tuple{}, ::Val{N_reductions}) where {N, N_reductions}
    ntuple(_-> first(composite), Val(N_reductions))
end

@inline function expand_parallel_composite(composite::NTuple{N, AbstractRheology}, funs_local, ::Val{N_reductions}) where {N, N_reductions}
    (ntuple(_-> first(composite), Val(N_reductions))..., expand_composite(composite, funs_local, parallel_state_functions)...)
end

@inline function expand_parallel_composite(composite::NTuple{N, AbstractRheology}, ::Tuple{}, ::Val{N_reductions}) where {N, N_reductions}
    ntuple(_-> first(composite), Val(N_reductions))
end

@inline function expand_composite(composite::NTuple{N, AbstractRheology}, ::Tuple{}, ::Val{N_reductions}) where {N, N_reductions}
    ntuple(_-> first(composite), Val(N_reductions))
end

@generated function expand_composite(composite::NTuple{N, AbstractRheology}, funs_local::NTuple{NF, Any}, fn::F) where {N, NF, F}
    quote
        @inline
        c = Base.@ntuple $N i ->  begin
            _expand_composite(composite[i], funs_local, fn(composite[i]))
        end
        Base.IteratorsMD.flatten(Base.IteratorsMD.flatten(c))
        # Base.IteratorsMD.flatten(c)
    end
end

_expand_composite(compositeᵢ, ::Val{true})          = ((compositeᵢ),)
_expand_composite(::AbstractRheology, ::Val{false}) = ()

@generated function _expand_composite(compositeᵢ, funs_local::NTuple{N1, Any}, fns::NTuple{N2, Any}) where {N1,N2}
    quote
        @inline
        Base.@ntuple $N2 i -> begin
            _expand_composite(compositeᵢ, isin_functions(fns[i], funs_local))
        end
    end
end

function isin_functions(fn::F,  fns::NTuple{N, Any}) where {F<:Function, N} 
    compare(fn, first(fns), Base.tail(fns), Val(false))
end

@inline _compare(::F1,  ::F2, ::Val{B}) where {F1, F2, B} = compare(Val(false), Val(B))
@inline _compare(::F,  ::F, ::Val{B})   where {F, B}      = compare(Val(true), Val(B))

@inline compare(::Val{false}, ::Val{false}) = Val(false)
@inline compare(::Val, ::Val) = Val(true)

@inline compare(fn::F1, fns₁::F2, fns::NTuple{N, Any}, ::Val{B}) where {F1, F2, N, B} = compare(fn, first(fns), Base.tail(fns), _compare(fn, fns₁, Val(B)))
@inline compare(fn::F1, fns₁::F2, ::Tuple{}, ::Val{B}) where {F1, F2, B} = _compare(fn, fns₁, Val(B))


@inline function split_series_composite(composite, unique_funs_global, funs_local)
    split_composite(composite, unique_funs_global, funs_local, series_state_functions)
end

@inline function split_parallel_composite(composite, unique_funs_global, funs_local)
    split_composite(composite, unique_funs_global, funs_local, parallel_state_functions)
end

@inline function split_composite(composite::NTuple{N, AbstractRheology}, unique_funs_global, funs_local, fn::F) where {N, F}
    (split_composite(composite, unique_funs_global, fn), expand_composite(composite, funs_local, fn))
end

@inline function split_composite(composite::NTuple{N, AbstractRheology}, unique_funs_global, ::Tuple{}, fn::F) where {N, F}
    (split_composite(composite, unique_funs_global, fn), ())
end

@generated function split_composite(composite::NTuple{N, AbstractRheology}, funs_global, fn::F) where {N,F}
    quote
        @inline
        c = Base.@ntuple $N i -> begin
            compositeᵢ = composite[i]
            fns = fn(compositeᵢ)
            _split_composite(compositeᵢ, funs_global, fns)
        end
        Base.IteratorsMD.flatten(c)
    end
end

@generated function _split_composite(compositeᵢ, funs_global, fns::NTuple{N, Any}) where N
    quote
        @inline
        Base.@nexprs $N i -> begin
            fns[i] ∈ funs_global && return (compositeᵢ,)
        end
        return ()
    end
end

function generate_args_state_functions(reduction_ind::NTuple{N, Any}, local_x, ::Val{N_reductions}) where {N, N_reductions}
    ntuple(Val(N_reductions)) do i
        @inline
        if i ≤ N
            ntuple(Val(length(reduction_ind[i]))) do j
                @inline
                @inbounds ind = reduction_ind[i][j]
                @inbounds local_x[ind]
            end
        else
            (zero(eltype(local_x)),)
        end
    end
end
 
@inline generate_args_state_functions(::Tuple{}, ::Any, ::Val) = ()

@inline getindex_tuple(x, inds::NTuple{N, Int}) where N = ntuple(i -> @inbounds(x[inds[i]]), Val(N))

function getindex_tuple(x, inds::NTuple{N, Int}) where N 
    ntuple(Val(N)) do i 
        @inline 
        @inbounds ind = inds[i]
        iszero(ind) ? zero(eltype(x)) : x[ind]
    end
end