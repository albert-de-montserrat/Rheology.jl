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

# table of methods needed per rheology
@inline state_functions(::LinearViscosity) = (compute_shear_strain,)
@inline state_functions(::PowerLawViscosity) = (compute_shear_strain,)
@inline state_functions(::Elasticity) = compute_shear_strain, compute_volumetric_strain
@inline state_functions(::DruckerPrager) = compute_shear_strain, compute_volumetric_strain, compute_lambda
@inline state_functions(::AbstractRheology) = error("Rheology not defined")
# handle tuples
@inline state_functions(r::NTuple{N, AbstractRheology}) where N = state_functions(first(r))..., state_functions(Base.tail(r))...
@inline state_functions(::Tuple{})= ()