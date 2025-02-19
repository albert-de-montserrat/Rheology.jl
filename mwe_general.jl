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

function eval_residual(x, composite, state_funs, subtractor_vars, inds_x_to_subtractor, inds_args_to_x, args_template, args_all)
    subtractor = generate_subtractor(x, inds_x_to_subtractor)
    args_tmp   = generate_args_from_x(x, inds_args_to_x, args_template, args_all)
    eval_state_functions(state_funs, composite, args_tmp) - subtractor - subtractor_vars
end

function eval_residual2(x, composite, state_funs, funs_global, subtractor_vars, inds_x_to_subtractor, inds_args_to_x, args_template, args_all, args_solve)
    subtractor = generate_subtractor(x, inds_x_to_subtractor)
    args_tmp   = generate_args_from_x(x, inds_args_to_x, args_template, args_all)
    args_solve2 = update_args2(args_solve, x)
    subtractor_global = generate_subtractor_global(composite, state_funs, funs_global, inds_x_to_subtractor, args_solve2)

    eval_state_functions(state_funs, composite, args_tmp) - subtractor - subtractor_vars + subtractor_global
end

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

getindex_tuple(x, inds::NTuple{N, Int}) where N = ntuple(i -> x[inds[i]], Val(N))
@generated function generate_subtractor(x::SVector{N, T}, inds::NTuple{N, Int}) where {N,T}
    quote
        Base.@nexprs $N i -> x_i = iszero(inds[i]) ? zero(T) : x[inds[i]]
        Base.@ncall $N SVector x
    end 
end

_generate_args_from_x(xᵢ::NTuple{N, Number}, ::Tuple{}, ::Any) where N = xᵢ
function _generate_args_from_x(xᵢ::NTuple{N, Number}, args_templateᵢ::NamedTuple, args_allᵢ::NamedTuple) where N
    tmp = (; zip(keys(args_templateᵢ), xᵢ)...)
    merge(args_allᵢ, tmp)
end

@generated function generate_args_from_x(x::SVector{N}, inds_args_to_x, args_template, args_all) where N
    quote
        @inline 
        Base.@ntuple $N i -> begin
            xᵢ = getindex_tuple(x, inds_args_to_x[i]) 
            _generate_args_from_x(xᵢ, args_template[i], args_all[i])
        end
    end
end

@generated function mapping_x_to_subtractor(state_funs::NTuple{N1, Any}, unique_funs_local::NTuple{N2, Any}) where {N1, N2}
    quote
        @inline 
        Base.@ntuple $N1 i -> begin
            ind = 0
            Base.@nexprs $N2 j -> begin
                check = unique_funs_local[j] === state_funs[i]
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
            fns = series_state_functions(composite[i])
            _expand_composite(composite[i], funs_local, fns)
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

@generated function generate_subtractor_global(composite, state_funs::NTuple{N1, Any}, funs_global, inds_x_to_subtractor::NTuple{N2, Int}, args_solve) where {N1, N2}
    quote
        @inline
        v = Base.@ntuple $N1 i -> begin
            val = zero(eltype(args_solve))
            if state_funs[i] === state_var_reduction
                fn = funs_global[i]
                Base.@nexprs $N2 j -> begin
                    if iszero(inds_x_to_subtractor[j]) 
                        val += fn(composite[j], args_solve)
                    end
                end
            end
            val
        end
        SVector{$N1}(v)
    end
end

_local_series_state_functions(::typeof(compute_strain_rate)) = ()
_local_series_state_functions(::typeof(compute_volumetric_strain_rate)) = ()
_local_series_state_functions(fn::F) where F<:Function = (fn,)

@generated function local_series_state_functions(funs::NTuple{N, Any}) where N
    quote
        @inline
        f = Base.@ntuple $N i -> _local_series_state_functions(funs[i])
        Base.IteratorsMD.flatten(f)
    end
end

_global_series_state_functions(fn::typeof(compute_strain_rate)) = (fn, )
_global_series_state_functions(fn::typeof(compute_volumetric_strain_rate)) = (fn, )
_global_series_state_functions(::F) where F<:Function = ()

@generated function global_series_state_functions(funs::NTuple{N, Any}) where N
    quote
        @inline
        f = Base.@ntuple $N i -> _global_series_state_functions(funs[i])
        Base.IteratorsMD.flatten(f)
    end
end

function main(composite, vars, args_solve, args_other)

    funs_all           = series_state_functions(composite)
    funs_global        = global_series_state_functions(funs_all)
    args_global        = all_differentiable_kwargs(funs_global)
    # vars_global        = vars
    # unique_funs_global = flatten_repeated_functions(funs_global)
    args_global_aug    = ntuple(Val(length(args_global))) do i 
        merge(args_global[i], args_other)
    end
    
    funs_local        = local_series_state_functions(funs_all)
    unique_funs_local = flatten_repeated_functions(funs_local)
    args_local        = all_differentiable_kwargs(funs_local)
    vars_local        = ntuple(Val(length(args_local))) do i 
        k = keys(args_local[i])
        v = (vars[first(k)],)
        (; zip(k, v)...)
    end
    args_local_aug = ntuple(Val(length(args_local))) do i 
        merge(args_local[i], args_other, vars)
    end

    N_reductions      = length(unique_funs_local)
    state_reductions  = ntuple(i -> state_var_reduction, Val(N_reductions))
    args_reduction    = ntuple(_ -> (), Val(N_reductions))
    state_funs        = merge_funs(state_reductions, funs_local)
    reduction_ind     = reduction_funs_args_indices(funs_local, unique_funs_local)

    N                 = length(state_funs)
    subtractor_vars   = SVector{N}(i ≤ N_reductions ? values(vars)[i] : 0e0 for i in 1:N)

    inds_args_to_x    = tuple(
        ntuple(i -> reduction_ind[i] .+ N_reductions, Val(N_reductions))...,
        ntuple(x -> (x + N_reductions,), Val(length(funs_local)))...,
    )

    inds_x_to_subtractor = mapping_x_to_subtractor(state_funs, unique_funs_local)

    local_x = SA[Base.IteratorsMD.flatten(values.(vars_local))...]
    state_x = SA[values(args_solve)...]
    x       = SA[state_x..., local_x...]

    args_state = ntuple(Val(N_reductions)) do i
        ntuple(Val(length(reduction_ind[i]))) do j
            ind = reduction_ind[i][j]
            local_x[ind]
        end
    end

    # 
    args_all      = tuple(args_state..., args_local_aug...)
    # template for args to do pattern matching later
    args_template = tuple(args_reduction..., args_local...)

    # need to expand the composite for the local equations
    composite_expanded = expand_composite(composite, funs_local, Val(N_reductions))

    R, J = value_and_jacobian(
        x -> eval_residual2(x, composite_expanded, state_funs, funs_global, subtractor_vars, inds_x_to_subtractor, inds_args_to_x, args_template, args_all, args_solve), 
        AutoForwardDiff(), 
        x
    )
    J \ R

end

viscous    = LinearViscosity(5e19)
powerlaw   = PowerLawViscosity(5e19, 3)
elastic    = Elasticity(1e10, 1e12) # im making up numbers
composite  = viscous, powerlaw
dt         = 1e10
vars       = (; ε = 1e-15,) # input variables
args_solve = (; τ = 1e2,) # we solve for this, initial guess
args_other = (; ) # other args that may be needed, non differentiable

# main((composite, vars, args_solve, args_other)...)
# @b main($(composite, vars, args_solve, args_other)...)
