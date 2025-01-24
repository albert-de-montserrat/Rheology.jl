using Test 
using StaticArrays
using ForwardDiff

abstract type AbstractRheology end

struct LinearViscosity{T} <: AbstractRheology
    η::T
end

struct PowerLawViscosity{T,I} <: AbstractRheology
    η::T
    n::I # DO NOT PROMOTE TO FP BY DEFAULT
end

struct Elasticity{T} <: AbstractRheology
    G::T
    K::T
end

## NOT NEEDED FOR NOW
# state_variables(::LinearViscosity) = (:stress,)
# state_variables(::PowerLawViscosity) = (:stress,)
# state_variables(::Elasticity) = :stress, :pressure
# state_variables(::AbstractRheology) = error("Rheology not defined")

@inline state_functions(::LinearViscosity) = (compute_shear_strain,)
@inline state_functions(::PowerLawViscosity) = (compute_shear_strain,)
@inline state_functions(::Elasticity) = compute_shear_strain, compute_volumetric_strain
@inline state_functions(::AbstractRheology) = error("Rheology not defined")
# handle tuples
@inline state_functions(r::NTuple{N, AbstractRheology}) where N = state_functions(first(r))..., state_functions(Base.tail(r))...
@inline state_functions(::Tuple{})= ()

# compute_shear_strain methods
@inline compute_shear_strain(r::LinearViscosity; τ = 0, kwargs...) = τ / (2 * r.η)
@inline compute_shear_strain(r::PowerLawViscosity; τ = 0, kwargs...) = τ^r.n / (2 * r.η)
@inline compute_shear_strain(r::Elasticity; τ = 0, τ0 = 0, dt =Inf, kwargs...) = (τ - τ0) / (2 * r.G * dt)
@inline compute_shear_strain(r::AbstractRheology; kwargs...) = 0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_shear_strain(r::AbstractRheology, kwargs::NamedTuple) = compute_shear_strain(r; kwargs...)

# compute_volumetric_strain methods
@inline compute_volumetric_strain(r::Elasticity; P=0, P0 = 0, dt =Inf, kwargs...) = (P - P0) / (2 * r.K * dt)
@inline compute_volumetric_strain(r::AbstractRheology; kwargs...) = 0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_volumetric_strain(r::AbstractRheology, kwargs::NamedTuple) = compute_volumetric_strain(r; kwargs...)

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

function get_unique_state_functions(composite::NTuple{N, AbstractRheology}) where N
    funs  = state_functions(composite)
    # get unique state functions
    return flatten_repeated_functions(funs)
end

### Scripting
# elemental rheologies
viscous  = LinearViscosity(1e20)
powerlaw = PowerLawViscosity(1e30, 2)
elastic  = Elasticity(1e10, Inf)
# define args
args = (; τ = 1e9, P = 1e9)
args2 = SA[values(args)...]
# composite rheology
composite = viscous, elastic, powerlaw
# pull state functions
statefuns = get_unique_state_functions(composite)

@code_llvm get_unique_state_functions0(composite)
@code_llvm get_unique_state_functions(composite)


# local jacobians
J1 = ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[1], (; τ = x[1], P = x[2])), args2)
J2 = ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[2], (; τ = x[1], P = x[2])), args2)
J3 = ForwardDiff.jacobian( x-> eval_state_functions(statefuns, composite[3], (; τ = x[1], P = x[2])), args2)


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
