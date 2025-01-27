abstract type AbstractRheology end
abstract type AbstractPlasticity <: AbstractRheology end # in case we need spacilization at some point

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

struct DruckerPrager{T} <: AbstractPlasticity
    C::T 
    ϕ::T # in degrees for now
    ψ::T # in degrees for now
end

DruckerPrager(args...) = DruckerPrager(promote(args...)...)

## METHODS FOR SERIES MODELS

# table of methods needed per rheology
@inline series_state_functions(::LinearViscosity) = (compute_shear_strain,)
@inline series_state_functions(::PowerLawViscosity) = (compute_shear_strain,)
@inline series_state_functions(::Elasticity) = compute_shear_strain, compute_volumetric_strain
@inline series_state_functions(::DruckerPrager) = compute_shear_strain, compute_volumetric_strain, compute_lambda
@inline series_state_functions(::AbstractRheology) = error("Rheology not defined")
# handle tuples
@inline series_state_functions(r::NTuple{N, AbstractRheology}) where N = series_state_functions(first(r))..., series_state_functions(Base.tail(r))...
@inline series_state_functions(::Tuple{})= ()

## METHODS FOR PARALLEL MODELS
# table of methods needed per rheology
@inline parallel_state_functions(::LinearViscosity) = (compute_stress,)
@inline parallel_state_functions(::PowerLawViscosity) = (compute_stress,)
@inline parallel_state_functions(::Elasticity) = compute_stress, compute_pressure
@inline parallel_state_functions(::DruckerPrager) = compute_stress, compute_pressure, compute_lambda
@inline parallel_state_functions(::AbstractRheology) = error("Rheology not defined")
# handle tuples
@inline parallel_state_functions(r::NTuple{N, AbstractRheology}) where N = parallel_state_functions(first(r))..., parallel_state_functions(Base.tail(r))...
@inline parallel_state_functions(::Tuple{})= ()