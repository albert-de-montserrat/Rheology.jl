using LinearAlgebra
using StaticArrays
using ForwardDiff

include("rheology_types.jl")
include("state_functions.jl")
# include("composite.jl") # not functional yet


@inline function augment_args(args, Δx)
    k = keys(args)
    vals = MVector(values(args))
    for i in eachindex(Δx)
        vals[i] += Δx[i]
    end
    return (; zip(k, vals)...)
end

function bt_line_search(Δx, J, R, statefuns, composite, args, vars; α=1.0, ρ=0.5, c=1e-4, α_min=1e-8)

    perturbed_args = augment_args(args, α * Δx)
    perturbed_R    = -SA[values(vars)...]

    # this will be put into a function (harcoded for now)
    Base.@nexprs 2 i -> begin
        @inline
        perturbed_R += eval_state_functions(statefuns, composite[i], perturbed_args)
    end

    while norm(perturbed_R) > norm(R + (c * α * (J * Δx)))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end

        perturbed_args = augment_args(args, α * Δx)
        perturbed_R    = -SA[values(vars)...]
    
        # this will be put into a function (harcoded for now)
        Base.@nexprs 2 i -> begin
            @inline
            perturbed_R += eval_state_functions(statefuns, composite[i], perturbed_args)
        end
        
        @show α
    end

    return α
end


function main_series(args; max_iter=100, tol=1e-10, verbose=false)
    viscous  = LinearViscosity(5e19)
    powerlaw = PowerLawViscosity(5e19, 3)
    # elastic  = Elasticity(1e10, 1e12) # im making up numbers
    # drucker  = DruckerPrager(1e6, 30, 10)
    # define args
    # dt = 1e10
    # args = (; τ = 1e9, P = 1e9, λ = 0e0) # we solve for this
    # args = (; τ = 1e2) # we solve for this, initial guess
    x    = SA[values(args)...]
    vars = (; ε = 1e-15) # input variables
    # composite rheology
    composite = viscous, powerlaw
    # pull state functions
    statefuns = get_unique_state_functions(composite, :series)

    ## START NEWTON RAPHSON SOLVER
    err, iter = 1e3, 0

    while err > tol
        iter += 1

        # compute the global jacobian and residual
        R = -SA[values(vars)...]
        J = @SMatrix zeros(length(statefuns), length(statefuns))
        Base.@nexprs 2 i -> begin # this will be put into a function, hardcoded for now
            J += ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[i], (; τ = x[1])), x)
            R += eval_state_functions(statefuns, composite[i], args)
        end
    
        Δx    = -J \ R
        α     = bt_line_search(Δx, J, R, statefuns, composite, args, vars)
        x    += α * Δx
        err   = norm(Δx/abs(x) for (Δx,x) in zip(Δx, x))

        iter > max_iter &&  break

        # update args, this should be generalized
        args = (; τ=x[1])

        if verbose; println("iter: $iter, x: $x, err: $err, α = $α"); end

    end
    
end

args = (; τ = 1e2) # we solve for this, initial guess

@b main_series($args; verbose = false)

