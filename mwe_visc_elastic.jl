using LinearAlgebra
using StaticArrays
using ForwardDiff
using DifferentiationInterface

include("src/composite.jl") # not functional yet
include("src/rheology_types.jl")
include("src/state_functions.jl")
include("src/kwargs.jl")
include("src/matrices.jl")

function bt_line_search(Δx, J, R, statefuns, composite::NTuple{N, Any}, args, vars; α=1.0, ρ=0.5, c=1e-4, α_min=1e-8) where N
    perturbed_args = augment_args(args, α * Δx)
    perturbed_R    = compute_residual(composite, statefuns, vars, perturbed_args)
      
    while sqrt(sum(perturbed_R.^2)) > sqrt(sum((R + (c * α * (J * Δx))).^2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_args = augment_args(args, α * Δx)
        perturbed_R = compute_residual(composite, statefuns, vars, perturbed_args) 
    end
    return α
end

function eval_residual(x, composite_expanded, composite_global, state_funs, unique_funs_global, subtractor_vars, inds_x_to_subtractor, inds_args_to_x, args_template, args_all, args_solve, args_other)
    subtractor        = generate_subtractor(x, inds_x_to_subtractor)
    args_tmp          = generate_args_from_x(x, inds_args_to_x, args_template, args_all)
    args_solve2       = merge(update_args2(args_solve, x), args_other)
    subtractor_global = generate_subtractor_global(composite_global, state_funs, unique_funs_global, args_solve2)
    
    eval_state_functions(state_funs, composite_expanded, args_tmp) - subtractor - subtractor_vars + subtractor_global
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

@inline getindex_tuple(x, inds::NTuple{N, Int}) where N = ntuple(i -> @inbounds(x[inds[i]]), Val(N))

function getindex_tuple(x, inds::NTuple{N, Int}) where N 
    ntuple(Val(N)) do i 
        @inline 
        @inbounds ind = inds[i]
        iszero(ind) ? zero(eltype(x)) : x[ind]
    end
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

@inline _generate_args_from_x(xᵢ::NTuple{N, Number}, ::Tuple{}, ::Any) where N = xᵢ

function _generate_args_from_x(xᵢ::NTuple{N, Number}, args_templateᵢ::NamedTuple, args_allᵢ::NamedTuple) where N
    tmp = (; zip(keys(args_templateᵢ), xᵢ)...)
    merge(args_allᵢ, tmp)
end

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

@generated function expand_composite(composite::NTuple{N, AbstractRheology}, funs_local) where N
    quote
        @inline
        c = Base.@ntuple $N i -> begin
            compositeᵢ = @inbounds composite[i]
            fns = series_state_functions(compositeᵢ)
            _expand_composite(compositeᵢ, funs_local, fns)
        end
        Base.IteratorsMD.flatten(Base.IteratorsMD.flatten(c))
    end
end

@generated function _expand_composite(compositeᵢ, funs_local, fns::NTuple{N, Any}) where N
    quote
        @inline
        Base.@ntuple $N i -> begin
            fns[i] ∈ funs_local ? ((compositeᵢ),) : ()
        end
    end
end

@inline function expand_composite(composite::NTuple{N, AbstractRheology}, funs_local, ::Val{N_reductions}) where {N, N_reductions}
    (ntuple(_-> first(composite), Val(N_reductions))..., expand_composite(composite, funs_local)...)
end


@inline function split_composite(composite::NTuple{N, AbstractRheology}, unique_funs_global, funs_local) where {N}
    (split_composite(composite, unique_funs_global), expand_composite(composite, funs_local))
end

@generated function split_composite(composite::NTuple{N, AbstractRheology}, funs_global) where N
    quote
        @inline
        c = Base.@ntuple $N i -> begin
            compositeᵢ = composite[i]
            fns = series_state_functions(compositeᵢ)
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

for fn in (:compute_strain_rate, :compute_volumetric_strain_rate)
    @eval _local_series_state_functions(::typeof($fn)) = ()
end
@inline _local_series_state_functions(fn::F) where F<:Function = (fn,)

@generated function local_series_state_functions(funs::NTuple{N, Any}) where N
    quote
        @inline
        f = Base.@ntuple $N i -> _local_series_state_functions(@inbounds(funs[i]))
        Base.IteratorsMD.flatten(f)
    end
end

for fun in (:compute_strain_rate, :compute_volumetric_strain_rate)
    @eval @inline _global_series_state_functions(fn::typeof($fun)) = (fn, )
end
@inline _global_series_state_functions(::F) where {F<:Function} = ()

@generated function global_series_state_functions(funs::NTuple{N, Any}) where N
    quote
        @inline
        f = Base.@ntuple $N i -> _global_series_state_functions(@inbounds(funs[i]))
        Base.IteratorsMD.flatten(f)
    end
end

function generate_indices_from_args_to_x(funs_local::NTuple{N, Any}, reduction_ind, ::Val{N_reductions}) where {N, N_reductions}
    inds_global = ntuple(Val(N_reductions)) do i 
        @inline
         i ≤ N ? reduction_ind[i] .+ N_reductions : (0,)
    end
    inds_local = ntuple(x -> (x + N_reductions,), Val(N))
    
    tuple(inds_global...,  inds_local...)
end

@inline generate_indices_from_args_to_x(::Tuple{}, ::Tuple{}, ::Val) = ()

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

@inline differentiable_kwargs(::Type{T}, ::typeof(state_var_reduction))                    where T = (; )

function main(composite, vars, args_solve0, args_other)

    funs_all           = series_state_functions(composite)
    funs_global        = global_series_state_functions(funs_all)
    args_solve         = merge(differentiable_kwargs(funs_global), args_solve0)
    args_global        = all_differentiable_kwargs(funs_global)
    # vars_global        = vars
    unique_funs_global = flatten_repeated_functions(funs_global)
    args_global_aug    = ntuple(Val(length(args_global))) do i 
        merge(args_global[i], args_other)
    end
    
    funs_local        = local_series_state_functions(funs_all)
    unique_funs_local = flatten_repeated_functions(funs_local)
    args_local        = all_differentiable_kwargs(funs_local)
    vars_local        = ntuple(Val(length(args_local))) do i 
        k = keys(args_local[i])
        v = (first(k) ∈ keys(vars) ? vars[first(k)] : 0e0,)
        (; zip(k, v)...)
    end
    args_local_aug = ntuple(Val(length(args_local))) do i 
        merge(args_local[i], args_other, vars)
    end

    # N_reductions         = length(unique_funs_local)
    N_reductions         = length(unique_funs_global)
    # N_reductions         = length(vars)
    state_reductions     = ntuple(i -> state_var_reduction, Val(N_reductions))
    args_reduction       = ntuple(_ -> (), Val(N_reductions))
    state_funs           = merge_funs(state_reductions, funs_local)

    # indices that map the local functions to the respective reduction of the state functions
    reduction_ind        = reduction_funs_args_indices(funs_local, unique_funs_local)

    N                    = length(state_funs)
    N_reductions0        = min(N_reductions, length(vars))  # to be checked
    subtractor_vars      = SVector{N}(i ≤ N_reductions0 ? values(vars)[i] : 0e0 for i in 1:N)

    inds_args_to_x       = generate_indices_from_args_to_x(funs_local, reduction_ind, Val(N_reductions))

    # mapping from the state functions to the subtractor
    inds_x_to_subtractor = mapping_x_to_subtractor(state_funs, unique_funs_local)

    local_x  = SA[Base.IteratorsMD.flatten(values.(vars_local))...]
    global_x = SA[values(args_solve)...]
    x        = SA[global_x..., local_x...]

    # this are the values from the local components that need to be sumed in the state functions, i.e. in compute_strain_rate / compute_volumetric_strain_rate
    args_state_reductions = generate_args_state_functions(reduction_ind, local_x, Val(N_reductions))
    
    # 
    args_all      = tuple(args_state_reductions..., args_local_aug...)
    # template for args to do pattern matching later
    args_template = tuple(args_reduction..., args_local...)

    # need to expand the composite for the local equations
    composite_expanded  = expand_composite(composite, funs_local, Val(N_reductions))
    composite_global, a = split_composite(composite, unique_funs_global, funs_local)

    R, J = value_and_jacobian(        
        x -> eval_residual(x, composite_expanded, composite_global, state_funs, unique_funs_global, subtractor_vars, inds_x_to_subtractor, inds_args_to_x, args_template, args_all, args_solve, args_other),
        AutoForwardDiff(), 
        x
    )
    # J \ R
    J
end

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
viscous1_s = LinearViscosityStress(5e19)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12) # im making up numbers
case       = :case8

composite, vars, args_solve0, args_other = if case === :case1 
    composite  = viscous1, powerlaw
    vars       = (; ε  = 1e-15) # input variables
    args_solve = (; τ  = 1e2  ) # we solve for this, initial guess
    args_other = (;) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case2
    composite  = viscous1, elastic
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2,   P = 1e6  ) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case3
    composite  = viscous1, elastic, powerlaw
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2,   P = 1e6  ) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case4
    composite  = viscous1, drucker
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2, ) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case5
    composite  = viscous1, elastic, powerlaw, drucker
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2,   P = 1e6,) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case6
    composite  = powerlaw, powerlaw
    vars       = (; ε  = 1e-15) # input variables
    args_solve = (; τ  = 1e2) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case7
    composite  = viscous1, viscous2
    vars       = (; ε  = 1e-15) # input variables
    args_solve = (; τ  = 1e2) # we solve for this, initial guess
    args_other = (; ) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case8
    composite  = viscous1, viscous2, drucker, drucker, elastic, powerlaw
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2,   P = 1e6,) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other
end

main(composite, vars, args_solve0, args_other)
@b main($(composite, vars, args_solve, args_other)...)
# @code_warntype main(composite, vars, args_solve0, args_other)