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

function main(vars, composite, args; mode = :series, max_iter=100, tol=1e-10, verbose=false)

    # pull state functions
    statefuns = get_unique_state_functions(composite, mode)

    # split args into differentiable and not differentiable
    args_diff, args_nondiff = split_args(args, statefuns)
    # rhs of the system of eqs, initial guess
    x = SA[values(args_diff)...]

    ## START NEWTON RAPHSON SOLVER
    max_iter = 100
    err, iter = 1e3, 0
    tol = 1e-8

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

    return update_args(args_diff, x)
end

# define rheologies
viscous  = LinearViscosity(1e6)
powerlaw = PowerLawViscosity(5e19, 3)
elastic  = Elasticity(1e10, 1e100) # im making up numbers
drucker  = DruckerPrager(1e6, 30, 10) # C, ϕ, ψ
# define args
dt = 1e10
# composite rheology
composite =  (viscous, drucker,) #, powerlaw, 
mode = :series

args = (; τ = 100e6, P=1e6, λ = 0, dt=dt) # we solve for this, initial guess
vars=input_vars = (; ε = 1e-15, θ = 0, λ = 0) # input variables

main(input_vars, composite, args; mode=mode, verbose = true)