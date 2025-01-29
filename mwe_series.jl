using LinearAlgebra
using StaticArrays
using ForwardDiff

include("rheology_types.jl")
include("state_functions.jl")
include("kwargs.jl")
include("matrices.jl")
# include("composite.jl") # not functional yet

function bt_line_search(Δx, J, R, statefuns, composite::NTuple{N, Any}, args, vars; α=1.0, ρ=0.5, c=1e-4, α_min=1e-8) where N
    perturbed_args = augment_args(args, α * Δx)
    perturbed_R = compute_residual(composite, statefuns, vars, perturbed_args)
      
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
end

viscous  = LinearViscosity(5e19)
powerlaw = PowerLawViscosity(5e19, 3)
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
main_series2(args; verbose = true)
#@b main_series2($args; verbose = false)

function main_series_viscoelastic(args; max_iter=100, tol=1e-10, verbose=false)

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
    
end

args = (; τ = 1e2, P = 1e6, dt = 1e10) # we solve for this, initial guess
main_series_viscoelastic(args; verbose = true)

x = @SMatrix rand(3,3);
#@b norm($x)
#@b sqrt(sum($x.^2))