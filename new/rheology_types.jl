abstract type AbstractRheology end
abstract type AbstractPlasticity <: AbstractRheology end # in case we need spacilization at some point

struct LinearViscosity{T} <: AbstractRheology
    η::T
end

# Linear viscous rheology for which we ony define a compute_stress routine
struct LinearViscosityStress{T} <: AbstractRheology
    η::T    
end

struct PowerLawViscosity{T,I} <: AbstractRheology
    η::T
    n::I # DO NOT PROMOTE TO FP BY DEFAULT
end

struct LTPViscosity{T,I} <: AbstractRheology
    η::T
    ε0::T
    Q::T
    σb::T
    σr::T
end

struct Elasticity{T} <: AbstractRheology
    G::T
    K::T
end

struct IncompressibleElasticity{T} <: AbstractRheology
    G::T
end

struct DruckerPrager{T} <: AbstractPlasticity
    C::T 
    ϕ::T # in degrees for now
    ψ::T # in degrees for now
end

DruckerPrager(args...) = DruckerPrager(promote(args...)...)

## METHODS FOR SERIES MODELS
@inline length_state_functions(r::AbstractRheology)                    = length(series_state_functions(r))
@inline length_state_functions(r::NTuple{N, AbstractRheology}) where N = length_state_functions(first(r))..., length_state_functions(Base.tail(r))...
@inline length_state_functions(r::Tuple{})                             = ()

# table of methods needed per rheology
@inline series_state_functions(::LinearViscosity)          = (compute_strain_rate,)
@inline series_state_functions(::LinearViscosityStress)    = (compute_stress,)
# @inline series_state_functions(::PowerLawViscosity)        = (compute_strain_rate,)
@inline series_state_functions(::PowerLawViscosity)        = (compute_stress,)
@inline series_state_functions(::Elasticity)               = compute_strain_rate, compute_volumetric_strain_rate
@inline series_state_functions(::IncompressibleElasticity) = (compute_strain_rate, )
@inline series_state_functions(::DruckerPrager)            = compute_strain_rate, compute_volumetric_strain_rate, compute_lambda
@inline series_state_functions(::DruckerPrager)            = (compute_strain_rate, compute_lambda)
#@inline series_state_functions(r::Series) = series_state_functions(r.elements)
@inline series_state_functions(::AbstractRheology)         = error("Rheology not defined")
# handle tuples
@inline series_state_functions(r::NTuple{N, AbstractRheology}) where N = series_state_functions(first(r))..., series_state_functions(Base.tail(r))...


# returns the flattened statefunctions along with NTuples with global & local element numbers
function series_state_functions(r::NTuple{N, AbstractRheology}, num::MVector{N,Int}) where N 
    statefuns     = (series_state_functions(first(r))...,  series_state_functions(Base.tail(r))...)
    len           = ntuple(i->length(series_state_functions(r[i])),N)
    statenum      = ntuple(i->val(i,len,num),Val(sum(len)))
    stateelements = ntuple(i->val_element(i,len),Val(sum(len)))

    return statefuns, statenum, stateelements
end

# does not allocate:
@inline series_state_functions(r::NTuple{N, AbstractRheology}) where N = series_state_functions(first(r))..., series_state_functions(Base.tail(r))...
@inline series_state_functions(::Tuple{})= ()

# function series_state_functions(composite::NTuple{N, AbstractRheology}) where N
#     statefuns = ntuple(Val(N)) do i
#         @inline 
#         series_state_functions(composite[i])
#     end
#     return statefuns    
# end


## METHODS FOR PARALLEL MODELS
# table of methods needed per rheology
@inline parallel_state_functions(::LinearViscosity) = (compute_stress,)
@inline parallel_state_functions(::PowerLawViscosity) = (compute_stress,)
@inline parallel_state_functions(::Elasticity) = compute_stress, compute_pressure
@inline parallel_state_functions(::IncompressibleElasticity) = (compute_stress, )
@inline parallel_state_functions(::DruckerPrager) = compute_stress, compute_pressure, compute_lambda, compute_plastic_strain_rate, compute_volumetric_plastic_strain_rate
@inline parallel_state_functions(::AbstractRheology) = error("Rheology not defined")


# handle tuples
@inline parallel_state_functions(r::NTuple{N, AbstractRheology}) where N = parallel_state_functions(first(r))..., parallel_state_functions(Base.tail(r))...
@inline parallel_state_functions(::Tuple{})= ()
# function parallel_state_functions(composite::NTuple{N, AbstractRheology}) where N
#     statefuns = ntuple(Val(N)) do j 
#         @inline 
#         parallel_state_functions(composite[j])
#     end
#     return statefuns    
# end


function parallel_state_functions(r::NTuple{N, AbstractRheology}, num::MVector{N,Int}) where N 
    statefuns = (parallel_state_functions(first(r))...,  parallel_state_functions(Base.tail(r))...)
    
    len = ntuple(i->length(parallel_state_functions(r[i])),N)
    statenum = ntuple(i->val(i,len,num),Val(sum(len)))
    stateelements = ntuple(i->val_element(i,len),Val(sum(len)))

    return statefuns, statenum, stateelements
end


@generated function flatten_repeated_functions(funs::NTuple{N, Any}) where {N}
    quote
        @inline 
        f = Base.@ntuple $N i -> i == 1 ? (funs[1],) : (funs[i] ∉ funs[1:i-1] ? (funs[i],) : ())
        Base.IteratorsMD.flatten(f)
    end
end

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
