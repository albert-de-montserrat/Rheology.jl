using Test 
using StaticArrays
using ForwardDiff

include("rheology_types.jl")
include("state_functions.jl")

# gives a tuple where its false if a state function is repeat (needs to be called only to generate the Jacobian)
# Note: this may start allocating; if so, try using recursion
@generated function get_state_fun_bool(funs::NTuple{N, Any}) where {N}
    quote
        @inline 
        Base.@ntuple $N i -> i == 1 ? true : (funs[i] ∉ funs[1:i-1] ? true : nothing)
    end
end

# peel functions to get the unique ones...
@inline peelfuns(fun::F, ::Bool) where F = (fun,)
@inline peelfuns(::F, ::Nothing) where F = ()
@inline peelfuns(funs::NTuple{N, Any}, bools::NTuple{N, Any}) where N = peelfuns(first(funs), first(bools))..., peelfuns(Base.tail(funs), Base.tail(bools))...
@inline peelfuns(::Tuple{}, ::Tuple{}) = () # not sure if this is needed

# function barrier to evaluate state function
eval_state_function(fn::F, r::AbstractRheology, args::NamedTuple) where F = fn(r, args)

@generated function eval_state_functions(funs::NTuple{N, Any}, r::AbstractRheology, args::NamedTuple) where {N}
    quote
        @inline 
        Base.@nexprs $N i -> x_i = eval_state_function(funs[i], r, args) 
        Base.@ncall $N SVector x
    end
end

# this is beautiful, gets compiled away
@generated function flatten_repeated_functions(funs::NTuple{N, Any}) where {N}
    quote
        @inline 
        f = Base.@ntuple $N i -> i == 1 ? (funs[1],) : (funs[i] ∉ funs[1:i-1] ? (funs[i],) : ())
        Base.IteratorsMD.flatten(f)
    end
end
function get_unique_state_functions0(composite::NTuple{N, AbstractRheology}) where N
    funs  = state_functions(composite)
    # state functions boolean
    funs_bool = get_state_fun_bool(funs)
    # get unique state functions
    return peelfuns(funs, funs_bool)
end

# function get_unique_state_functions(composite::NTuple{N, AbstractRheology}) where N
#     funs  = state_functions(composite)
#     # get unique state functions
#     return flatten_repeated_functions(funs)
# end

function get_unique_state_functions(composite::NTuple{N, AbstractRheology}, model::Symbol) where N
    funs = if model === :series
        get_unique_state_functions(composite, series_state_functions)
    elseif model === :parallel
        get_unique_state_functions(composite, parallel_state_functions)
    else
        error("Model not defined. Accepted models are :series or :parallel")
    end
    return funs
end

function get_unique_state_functions(composite::NTuple{N, AbstractRheology}, state_fn) where N
    funs  = state_fn(composite)
    # get unique state functions
    return flatten_repeated_functions(funs)
end


# elemental rheologies
function main_series()
    viscous  = LinearViscosity(1e20)
    powerlaw = PowerLawViscosity(1e30, 2)
    elastic  = Elasticity(1e10, 1e12) # im making up numbers
    drucker  = DruckerPrager(1e6, 30, 10)
    # define args
    dt = 1e10
    args = (; τ = 1e9, P = 1e9, λ = 0e0) # we solve for this
    args2 = SA[values(args)...]
    vars = (; ε = 1e-15, θ = 1e-20, λ = 0e0)
    # composite rheology
    composite = viscous, elastic, powerlaw, drucker
    # pull state functions
    statefuns = get_unique_state_functions(composite, :series)

    # compute the global jacobian
    J = @SMatrix zeros(length(statefuns), length(statefuns))
    Base.@nexprs 4 i -> begin
        J += ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[i], (; τ = x[1], P = x[2], λ = x[3], dt = dt)), args2)
    end
    J
end

# # function main_parallel()
#     viscous  = LinearViscosity(1e20)
#     powerlaw = PowerLawViscosity(1e20, 3)
#     elastic  = Elasticity(1e10, 1e12) # im making up numbers
#     drucker  = DruckerPrager(1e6, 30, 10)
#     # define args
#     dt = 1e10
#     # args = (; ε = 1e-15, θ = 1e-19) # we solve for this
#     # args_res = (; ε = 1e-15, θ = 1e-19, dt = dt) # we solve for this
#     args_res = (; ε = 1e-15, dt = dt) # we solve for this
#     args = (; ε = 1e-15) # we solve for this
#     args2 = SA[values(args)...]
#     x = MVector(args2)
#     # composite rheology
#     composite = viscous, powerlaw
#     # pull state functions
#     statefuns = get_unique_state_functions(composite, :parallel)
#     N = length(statefuns)

#     # allocate Jacobian and residual
#     JM = @MMatrix zeros(N, N)
#     RM = -MVector(args2)
#     Base.@nexprs 2 i -> begin
#         JM .+= ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[i], (; ε = x[1], θ = x[2], dt = dt)), args2)
#         RM .+= eval_state_functions(statefuns, composite[i], args_res)
#     end
#     J     = SMatrix(JM)
#     R     = SVector(RM)

#     Δx    = J \ R
#     α     = bt_line_search(x, Δx, J, R, args_res)
#     x   .+= α * Δx
#     err   = norm(Δx/abs(x) for (Δx,x) in zip(Δx, x))

#     # @test J[1,1] == 2 * viscous.η + 2 * elastic.G * dt
#     # @test J[2,2] == elastic.K * dt

#     # return J
# # end

# function main_parallel()
viscous  = LinearViscosity(1e20)
powerlaw = PowerLawViscosity(1e20, 3)
elastic  = Elasticity(1e10, 1e12) # im making up numbers
drucker  = DruckerPrager(1e6, 30, 10)
# define args
dt = 1e10
# args = (; ε = 1e-15, θ = 1e-19) # we solve for this
# args_res = (; ε = 1e-15, θ = 1e-19, dt = dt) # we solve for this
args_res = (; ε = 1e-15, dt = dt) # we solve for this
args = (; ε = 1e-15) # we solve for this
args2 = SA[values(args)...]
x = MVector(args2)
# composite rheology
composite = viscous, powerlaw
# pull state functions
statefuns = get_unique_state_functions(composite, :parallel)
N = length(statefuns)

# allocate Jacobian and residual
JM = @MMatrix zeros(N, N)
RM = @MVector zeros(N) # bad initial guess


Base.@nexprs 2 i -> begin
    JM .+= ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[i], (; ε = x[1], dt = dt)), args2)
    RM .+= eval_state_functions(statefuns, composite[i], args_res)
end
J     = SMatrix(JM)
R     = SVector(RM)

Δx    = J \ R
α     = bt_line_search(x, Δx, J, R, args_res)
x   .+= α * Δx
err   = norm(Δx/abs(x) for (Δx,x) in zip(Δx, x))

args2 = x

# @test J[1,1] == 2 * viscous.η + 2 * elastic.G * dt
# @test J[2,2] == elastic.K * dt

# return J
# end

# main_parallel()

@inline function augment_args(args, Δx)
    k = keys(args)
    vals = MVector(values(args))
    for i in 1:length(vals)-1
        vals[i] += Δx[i]
    end
    return (; zip(k, vals)...)
end

function bt_line_search(x, Δx, J, R, args; α=1.0, ρ=0.5, c=1e-4, α_min=1e-8)

    perturbed_args = augment_args(args, Δx)
    perturbed_RM   = copy(x)

    # this will be put into a function (harcoded for now)
    Base.@nexprs 2 i -> begin
        @inline
        perturbed_RM += eval_state_functions(statefuns, composite[i], perturbed_args)
    end

    while norm(perturbed_R) > norm(R + c * α * (J *Δx))
        α *= ρ
        α ≥ α_min && continue
        α = α_min
        break
    end

    return α
end



1
# ### Scripting
# # elemental rheologies
# viscous  = LinearViscosity(1e20)
# powerlaw = PowerLawViscosity(1e30, 2)
# elastic  = Elasticity(1e10, 1e12) # im making up numbers
# drucker  = DruckerPrager(1e6, 30, 10)
# # define args
# dt = 1e10
# args = (; τ = 1e9, P = 1e9, λ = 0)
# args2 = SA[values(args)...]
# # composite rheology
# composite = viscous, elastic, powerlaw, drucker
# # pull state functions
# statefuns = get_unique_state_functions(composite)

# # local jacobians
# J1 = ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[1], (; τ = x[1], P = x[2], λ = x[3], dt = dt)), args2)
# J2 = ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[2], (; τ = x[1], P = x[2], λ = x[3], dt = dt)), args2)
# J3 = ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[3], (; τ = x[1], P = x[2], λ = x[3], dt = dt)), args2)
# J4 = ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[4], (; τ = x[1], P = x[2], λ = x[3], dt = dt)), args2)

# # compute the global jacobian
# J = @SMatrix zeros(length(statefuns), length(statefuns))
# for c in composite
#     J += ForwardDiff.jacobian( x-> eval_state_functions(statefuns, c, (; τ = x[1], P = x[2], λ = x[3], dt = dt)), args2)
# end

### Tests
# viscous  = LinearViscosity(1e20)
# powerlaw = PowerLawViscosity(1e30, 2)
# elastic  = Elasticity(1e10, Inf)

# @test compute_shear_strain(viscous, (τ = 1e9)) == 5e-12
# @test compute_shear_strain(powerlaw, (τ = 1e9)) == 5e-13
# @test compute_shear_strain(elastic, (τ = 1e9, dt = 1e10)) == 5e-12

# @test compute_volumetric_strain(viscous, (;)) == 0
# @test compute_volumetric_strain(powerlaw, (;)) == 0
# @test compute_volumetric_strain(elastic, (P = 1e9, dt = 1e10)) == 0
