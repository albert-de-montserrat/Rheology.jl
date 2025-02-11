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
@inline series_state_functions(::LinearViscosity) = (compute_strain_rate,)
@inline series_state_functions(::PowerLawViscosity) = (compute_strain_rate,)
@inline series_state_functions(::Elasticity) = compute_strain_rate, compute_volumetric_strain_rate
@inline series_state_functions(::DruckerPrager) = compute_strain_rate, compute_volumetric_strain_rate, compute_lambda
@inline series_state_functions(::DruckerPrager) = (compute_strain_rate, compute_lambda)
#@inline series_state_functions(r::Series) = series_state_functions(r.elements)


@inline series_state_functions(::AbstractRheology) = error("Rheology not defined")
# handle tuples
@inline series_state_functions(r::NTuple{N, AbstractRheology}) where N = series_state_functions(first(r))..., series_state_functions(Base.tail(r))...
  

@inline function series_state_functions(r::NTuple{N, AbstractRheology}, number::Tuple) where N 
    # this can likely be improved. What it does is to give back a list with computational
    # functions for the whole series elements. Each element has a strain rate function, 
    # but if we have, say, different plastic rheologies each of them must have its 
    # own lambda function. This function guarantees that, along with a unique number for 
    # each of the additional entries

    # NOTE: can likely be improved w.r.t. allocations

    rr = series_state_functions.(r) # functions for all elements
    
    # Now create a unique list of functions along with their number
    rrr =  series_state_functions(r) 
    num = zeros(Int, length(rrr))
    n = 0
    for i=1:N
        for j=1:length(rr[i])
            n += 1
            if isa(number[i], Int)
                num[n] = number[i]
            else
                num[n] = number[i][1]
            end


        end
    end
    # remove the ones that are called "compute_strain_rate", but keep the rest

    ind = findall(rrr .!= compute_strain_rate.&& rrr .!= compute_volumetric_strain_rate  )
    
    out_state_funs = (compute_strain_rate, rrr[ind]...)
    out_number     = (0,num[ind]...,)   # 0 implies that all elements have this defined
    if any(rrr .== compute_volumetric_strain_rate)
        out_state_funs = (compute_strain_rate, compute_volumetric_strain_rate, rrr[ind]...)
        out_number     = (0,0, num[ind]...,)   # unsure if we can always assume  
    end
    
    return out_state_funs, out_number
end

@inline function series_state_functions1(r::SeriesModel{N}, numel::NTuple{N,Any}) where N 
    numfuns = ()
    statefuns = ()
    for i=1:N
        fns, nums = series_state_functions1(r[i], numel[i])
        statefuns = (statefuns..., fns...)
        numfuns = (numfuns..., nums...)
    end
    
    # At this stage we have a flattened list with all elements
    # It also includes "compute_strain_rate" and "compute_volumetric_strain_rate"
    # which need to be defined for all 
    isvolum = any(statefuns .== compute_volumetric_strain_rate)
    indall = findall((statefuns .!= compute_strain_rate) .&& (statefuns .!= compute_volumetric_strain_rate))
    if isvolum
        statefuns = (compute_strain_rate, compute_volumetric_strain_rate, statefuns[indall]...)
        numfuns = (0,0, numfuns[indall]...)
    else
        statefuns = (compute_strain_rate, statefuns[indall]...)
        numfuns = (0, numfuns[indall]...)
    end

    return statefuns, numfuns
end

@inline function series_state_functions1(r::ParallelModel{N}, numel::Tuple{Int,NTuple{N,Any}}) where N 
    statefuns, numfuns = series_state_functions1(r, numel[1])
    for i=1:N
        fns, nums = series_state_functions1(r[i], numel[2][i])
        statefuns = (statefuns..., fns...)
        numfuns = (numfuns..., nums...)
    end
    return statefuns, numfuns
end

@inline function series_state_functions1(r::AbstractRheology, number::Int64)
    out = series_state_functions(r)
    return out,  (fill(number,length(out))...,)
end

@inline series_state_functions(::Tuple{})= ()

## METHODS FOR PARALLEL MODELS
# table of methods needed per rheology
@inline parallel_state_functions(::LinearViscosity) = (compute_stress,)
@inline parallel_state_functions(::PowerLawViscosity) = (compute_stress,)
@inline parallel_state_functions(::Elasticity) = compute_stress, compute_pressure
@inline parallel_state_functions(::DruckerPrager) = compute_stress, compute_pressure, compute_lambda, compute_plastic_strain_rate, compute_volumetric_plastic_strain_rate
@inline parallel_state_functions(::AbstractRheology) = error("Rheology not defined")


# handle tuples
@inline parallel_state_functions(r::NTuple{N, AbstractRheology}) where N = parallel_state_functions(first(r))..., parallel_state_functions(Base.tail(r))...
@inline parallel_state_functions(::Tuple{})= ()


@inline function parallel_state_functions(r::NTuple{N, AbstractRheology}, number::NTuple{N, Int}) where N 
   
    # NOTE: can likely be improved w.r.t. allocations

    rr  = parallel_state_functions.(r) # functions for all elements
    
    # Now create a unique list of functions along with their number
    rrr =  parallel_state_functions(r) 
    num = zeros(Int, length(rrr))
    n = 0
    for i=1:N
        for j=1:length(rr[i])
            n += 1
            num[n] = number[i]
        end
    end
    # remove the ones that are called "compute_stress" and/or "compute_pressure", but keep the rest

    ind = findall(rrr .!= compute_stress .&& rrr .!= compute_pressure)

    out_state_funs = (compute_stress, rrr[ind]...)
    out_number     = (0,num[ind]...,)   # 0 implies that all elements have this defined
    if any(rrr .== compute_pressure)
        out_state_funs = (compute_stress, compute_pressure, rrr[ind]...)
        out_number     = (0,0, num[ind]...,)   # unsure if we can always assume  
    end
    
    return out_state_funs, out_number
end

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