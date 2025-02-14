using LinearAlgebra
using StaticArrays
using ForwardDiff

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

function main_series(vars, composite, args; max_iter=100, tol=1e-10, verbose=false)
    
    # pull state functions
    statefuns = get_unique_state_functions(composite, :series)

    # split args into differentiable and not differentiable
    args_diff, args_nondiff = split_args(args, statefuns)

    # rhs of the system of eqs, initial guess
    x = SA[values(args_diff)...]
    
    ## START NEWTON RAPHSON SOLVER
    err, iter = 1e3, 0

    while err > tol
        iter += 1

        # compute the global jacobian and residual
        J = compute_jacobian(x, composite, statefuns, args_diff, args_nondiff)
        R = compute_residual(composite, statefuns, vars, args)
    
        Δx    = -J \ R
        α     = bt_line_search(Δx, J, R, statefuns, composite, args, vars)
        x    += Δx * α
        err   = norm(Δx/abs(x) for (Δx,x) in zip(Δx, x))

        iter > max_iter &&  break

        # update args, this should be generalized
        # args = (; τ=x[1])
        args = update_args(args, x)

        if verbose; println("iter: $iter, x: $x, err: $err, α = $α"); end

    end
    return update_args(args_diff, x)
end

viscous  = LinearViscosity(5e19)
# powerlaw = PowerLawViscosity(5e19, 3)
# elastic  = Elasticity(1e10, 1e12) # im making up numbers
# drucker  = DruckerPrager(1e6, 30, 10)
# define args
# dt = 1e10
# args = (; τ = 1e9, P = 1e9, λ = 0e0) # we solve for this
# args = (; τ = 1e2) # we solve for this, initial guess
vars = (; ε = 1e-15) # input variables
# composite rheology
composite = viscous, powerlaw

# @b main_series($args; verbose = false)

args = (; τ = 1e2) # we solve for this, initial guess
main_series(vars, composite, args; verbose = true)

function main_series2(args; max_iter=100, tol=1e-10, verbose=false)
    viscous  = LinearViscosity(5e19)
    powerlaw = PowerLawViscosity(5e19, 3)
    vars = (; ε = 1e-15) # input variables
    # composite rheology
    composite = viscous, powerlaw
    # pull state functions
    statefuns = get_unique_state_functions(composite, :series)

    # split args into differentiable and not differentiable
    args_diff, args_nondiff = split_args(args, statefuns)
    # rhs of the system of eqs, initial guess
    x = SA[values(args_diff)...]

    ## START NEWTON RAPHSON SOLVER
    err, iter = 1e3, 0

    while err > tol
        iter += 1

        J = compute_jacobian(x, composite, statefuns, args_diff, args_nondiff)
        R = compute_residual(composite, statefuns, vars, args)

        Δx   = -J \ R
        α    = bt_line_search(Δx, J, R, statefuns, composite, args, vars)
        x   += Δx # * α
        err  = sqrt(sum( (Δx ./ abs.(x)).^2))
        iter > max_iter &&  break

        # update args, this should be generalized (still not general enough)
        args = update_args(args, x)

        if verbose; println("iter: $iter, x: $x, err: $err, α = $α"); end

    end
    
end

args = (; τ = 1e2) # we solve for this, initial guess
@code_warntype main_series2(args; verbose = true)

function main_series3(composite, args, vars, mode; max_iter=100, tol=1e-10, verbose=false)
    # pull state functions
    statefuns = get_unique_state_functions(composite, mode)

    # split args into differentiable and not differentiable
    args_diff, args_nondiff = split_args(args, statefuns)
    # rhs of the system of eqs, initial guess
    x = SA[values(args_diff)...]

    ## START NEWTON RAPHSON SOLVER
    err, iter = 1e3, 0

    while err > tol
        iter += 1

        J    = compute_jacobian(x, composite, statefuns, args_diff, args_nondiff)
        R    = compute_residual(composite, statefuns, vars, args)

        Δx   = -J \ R
        α    = bt_line_search(Δx, J, R, statefuns, composite, args, vars)
        x   += Δx # * α
        err  = sqrt(sum( (Δx ./ abs.(x)).^2))
        iter > max_iter &&  break

        # update args, this should be generalized (still not general enough)
        args = update_args(args, x)

        if verbose; println("iter: $iter, x: $x, err: $err, α = $α"); end

    end
end
@code_warntype main_series3(composite, args, vars, :series; verbose = true)

viscous  = LinearViscosity(5e19)
powerlaw = PowerLawViscosity(5e19, 3)
vars     = (; ε = 1e-15) # input variables

# composite rheology
composite = viscous, powerlaw
args      = (; τ = 1e2) # we solve for this, initial guess


viscous  = LinearViscosity(1e18)
powerlaw = PowerLawViscosity(5e19, 3)
elastic  = Elasticity(1e10, 1e100) # im making up numbers
drucker  = DruckerPrager(1e6, 30, 0) # C, ϕ, ψ
# define args
dt = 1e10
# composite rheology
composite =  (viscous, drucker,) #, powerlaw, 
mode = :series

@code_warntype main_series2(args; verbose = true)

#@b main_series2($args; verbose = false)

# function main_series_viscoelastic(args; max_iter=100, tol=1e-10, verbose=true)

    # define rheologies
    viscous  = LinearViscosity(5e19)
    powerlaw = PowerLawViscosity(5e19, 3)
    elastic  = Elasticity(1e10, 1e12) # im making up numbers
    # drucker  = DruckerPrager(1e6, 30, 10)
    # define args
    dt = 1e10
    vars = (; ε = 1e-15, θ = 1e-20) # input variables
    # composite rheology
    composite = viscous, powerlaw, elastic
    # pull state functions
    statefuns = get_unique_state_functions(composite, :series)

    # split args into differentiable and not differentiable
    args_diff, args_nondiff = split_args(args, statefuns)
    # rhs of the system of eqs, initial guess
    x = SA[values(args_diff)...]

    ## START NEWTON RAPHSON SOLVER
    err, iter = 1e3, 0

    while err > tol
        iter += 1

        J = compute_jacobian(x, composite, statefuns, args_diff, args_nondiff)
        R = compute_residual(composite, statefuns, vars, args)
    
        Δx  = -J \ R
        α   = bt_line_search(Δx, J, R, statefuns, composite, args, vars)
        x  += Δx * α
        err = norm(Δx/abs(x) for (Δx,x) in zip(Δx, x))

        iter > max_iter &&  break

        # update args, this should be generalized
        args = update_args(args, x)

        if verbose; println("iter: $iter, x: $x, err: $err, α = $α"); end

    end
    
# end

args = (; τ = 1e2, P = 1e6, dt = 1e10) # we solve for this, initial guess
main_series_viscoelastic(args; verbose = true)

x = @SMatrix rand(3,3);

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

viscous    = LinearViscosity(5e19)
powerlaw   = PowerLawViscosity(5e19, 3)
elastic    = Elasticity(1e10, 1e12) # im making up numbers
dt         = 1e10
vars       = (; ε = 1e-15, θ = 1e-20) # input variables
args_solve = (; τ = 1e2, P = 1e6) # we solve for this, initial guess
args_other = (; dt = 1e10) # other args that may be needed, non differentiable


funs_local     = parallel_state_functions(composite)
args_local     = all_differentiable_kwargs(funs_local)
args_local_aug = ntuple(Val(length(args_local))) do i 
    merge(args_local0[i], args_other)
end

unique_funs_local = flatten_repeated_functions(funs_local)
N_reductions      = length(unique_funs_local)
state_reductions  = ntuple(i -> state_var_reduction, Val(N_reductions))
args_reduction    = ntuple(_ -> (), Val(N_reductions))
state_funs        = merge_funs(state_reductions, funs_local)
reduction_ind     = reduction_funs_args_indices(funs_local, unique_funs_local)

inds_args_to_x    = tuple(
    tuple([ind .+ N_reductions for ind in reduction_ind]...)...,
    ntuple(x -> (x + N_reductions,), Val(length(funs_local)))...,
)

# # split args into differentiable and not differentiable
# args_diff, args_nondiff = split_args(args, statefuns)
# # rhs of the system of eqs, initial guess
# x = SA[values(args_diff)...]

local_x = SA[Base.IteratorsMD.flatten(values.(args_local))...]
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

i = 1
compositeᵢ = composite[i]
f = x -> begin
    args_tmp = generate_args_from_x(x, inds_args_to_x, args_template, args_all)
    eval_state_functions(state_funs, compositeᵢ, args_tmp)
end
ForwardDiff.jacobian(x -> f(x), x)

