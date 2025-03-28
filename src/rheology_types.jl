
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

# mark compressible rheologies
iscompressible(::AbstractRheology) = false
iscompressible(::Elasticity) = true

# This appears to allocate in certain combinations on 1.10:
iscompressible(::DruckerPrager{T}) where T = true

#=
function iscompressible(r::DruckerPrager{T}) where T
    iscomp = false
    if iszero(r.ψ)
        iscomp =  true
    end
    return iscomp
end
=#
# function iscompressible(r::NTuple{N, AbstractRheology}) where N
#     iscomp = false
#     for i=1:N
#         #@show el
#         el::AbstractRheology = r[i]
#         if iscompressible(el)
#             iscomp = true
#             break
#         end
#     end
#     return iscomp
# end

@generated function iscompressible(r::NTuple{N, AbstractRheology}) where N
    quote 
        @inline 
        Base.@nexprs $N i-> iscompressible(r[i]) && return true
        return false
    end
end

# mark shear rheologies
isshear(::AbstractRheology) = true      # for now all are true
#isshear(r::NTuple{N, AbstractRheology}) where N = any(isshear.(r))     # allocates sometimes

@generated function isshear(r::NTuple{N, AbstractRheology}) where N
    quote 
        @inline 
        Base.@nexprs $N i-> isshear(r[i]) && return true
        return false
    end
end


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


# Global number
function val(i::Int64,len::NTuple{N,Int},num::MVector{N,Int}) where N
    n,v = 1, 0
    for j=1:N, k=1:len[j]
        if n == i
            v = num[j]
        end
        n += 1
    end
    return v
end

# local number
function val_element(i::Int64,len::NTuple{N,Int}) where N
    n,v = 1, 0
    for j=1:N, k=1:len[j]
        if n == i
            v = j
        end
        n += 1
    end
    return v
end



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



#####
# helper functions

# Determine if a rheology requires volumetric deformation
isvolumetric(::AbstractRheology) = false
isvolumetric(::Elasticity) = true           # we can later add a case that is false if ν==0.5 
function isvolumetric(r::DruckerPrager) 
    if r.ψ == 0
        return false
    else
        return true
    end
end
# isvolumetric(r::SeriesModel) = any(isvolumetric.(r.children))
# isvolumetric(r::ParallelModel) = any(isvolumetric.(r.siblings))

#####

# gives a tuple where its false if a state function is repeat (needs to be called only to generate the Jacobian)
# Note: this may start allocating; if so, try using recursion
@generated function get_state_fun_bool(funs::NTuple{N, Any}) where {N}
    quote
        @inline 
        Base.@ntuple $N i -> i == 1 ? true : (funs[i] ∉ funs[1:i-1] ? true : nothing)
    end
end

# function barrier to evaluate state function
eval_state_function(fn::F, r::AbstractRheology, args) where F = fn(r, args)

@generated function eval_state_functions(funs::NTuple{N, Any}, r::AbstractRheology, args) where {N}
    quote
        @inline 
        Base.@nexprs $N i -> x_i = eval_state_function(funs[i], r, args[i]) 
        Base.@ncall $N SVector x
    end
end

@generated function eval_state_functions(funs::NTuple{N, Any}, r::NTuple{N, AbstractRheology}, args) where {N}
    quote
        @inline 
        Base.@nexprs $N i -> x_i = eval_state_function(funs[i], r[i], args[i]) 
        Base.@ncall $N SVector x
    end
end

@inline eval_state_functions(::Tuple{Vararg{Any, N}}, ::NTuple{N, AbstractRheology}, ::@NamedTuple{}) where N = @SVector zeros(N)


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

# get_unique_state_functions(::AbstractCompositeModel, ::Any) = ()