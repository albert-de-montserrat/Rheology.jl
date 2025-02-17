using LinearAlgebra
using StaticArrays
using ForwardDiff
using DifferentiationInterface

include("composite.jl") # not functional yet
include("rheology_types.jl")
include("state_functions.jl")
include("kwargs.jl")
include("matrices.jl")

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
            fns = parallel_state_functions(composite[i])
            _expand_composite(composite[i], funs_local, fns)
        end
        Base.IteratorsMD.flatten(c)
    end
end

@generated function _expand_composite(compositeᵢ, funs_local, fns::NTuple{N, Any}) where N
    quote
        @inline
        Base.@ntuple $N i -> begin
            fns[i] ∈ funs_local ? compositeᵢ : ()
        end
    end
end

function main(composite, vars, args_solve, args_other)

    funs_local     = parallel_state_functions(composite)
    args_local     = all_differentiable_kwargs(funs_local)
    vars_local     = ntuple(Val(length(args_local))) do i 
        k = keys(args_local[i])
        v = (vars[first(k)],)
        (; zip(k, v)...)
    end
    args_local_aug = ntuple(Val(length(args_local))) do i 
        merge(args_local[i], args_other, vars)
    end

    unique_funs_local = flatten_repeated_functions(funs_local)
    N_reductions      = length(unique_funs_local)
    state_reductions  = ntuple(i -> state_var_reduction, Val(N_reductions))
    args_reduction    = ntuple(_ -> (), Val(N_reductions))
    state_funs        = merge_funs(state_reductions, funs_local)
    reduction_ind     = reduction_funs_args_indices(funs_local, unique_funs_local)

    N                 = length(state_funs)
    subtractor_vars   = SVector{N}(i ≤ N_reductions ? values(vars)[i] : 0e0 for i in 1:N)

    # args_residual     = residual_kwargs(state_funs)

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

    args_all      = tuple(args_state..., args_local_aug...)
    args_template = tuple(args_reduction..., args_local...)
    # args_tmp      = generate_args_from_x(x, inds_args_to_x, args_template, args_all)

    # this is hardcoded for now...
    dummy_composite = ntuple(_-> first(composite), Val(N_reductions))
    composite2      = (dummy_composite..., expand_composite(composite, funs_local)...)

    R, J = value_and_jacobian(
        x -> eval_residual(x, composite2, state_funs, subtractor_vars, inds_x_to_subtractor, inds_args_to_x, args_template, args_all), 
        AutoForwardDiff(), 
        x
    )
    J \ R

end

viscous    = LinearViscosity(5e19)
powerlaw   = PowerLawViscosity(5e19, 3)
elastic    = Elasticity(1e10, 1e12) # im making up numbers
composite  = viscous ,powerlaw ,elastic
dt         = 1e10
vars       = (; ε = 1e-15, θ = 1e-20) # input variables
args_solve = (; τ = 1e2, P = 1e6) # we solve for this, initial guess
args_other = (; dt = 1e10) # other args that may be needed, non differentiable

main((composite, vars, args_solve, args_other)...)
