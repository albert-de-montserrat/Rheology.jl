abstract type AbstractRheology end
abstract type AbstractPlasticity <: AbstractRheology end # in case we need spacilization at some point

abstract type AbstractCompositeModel <: AbstractRheology  end

@inline series_state_functions(::AbstractCompositeModel)= ()
@inline parallel_state_functions(::AbstractCompositeModel)= ()

#struct CompositeModel{Nstrain, Nstress, T, NUM} <: AbstractCompositeModel
struct CompositeModel{Nstrain, Nstress, T} <: AbstractCompositeModel

    components::T
   # number::NUM

   
end
function CompositeModel(composite::T) where T
    Nstrain = number_strain_rate_components(composite)
    Nstress = number_stress_components(composite)
    #Num     = number_elements(composite)
    #new{Nstrain, Nstress, T, typeof(Num)}(composite, Num)
    return CompositeModel{Nstrain, Nstress, T}(composite)
end
CompositeModel(x::Vararg{Any, N}) where N = CompositeModel(tuple(x)...)

struct SeriesModel{N, T, F} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    children::T # vertical stacking
    funs::F
    num::MVector{N, Int}
end


function SeriesModel(composite::T) where T
    funs = get_unique_state_functions(composite, :series)
    funs_flat = flatten_repeated_functions(funs)
    N = length(composite)
    SeriesModel{N, T, typeof(funs_flat)}(composite, funs_flat,  MVector{N,Int}(1:N))
end
#update_numbers(s::SeriesModel{N, T, F}, num::NTuple) where {N, T, F} = SeriesModel{N, T, F}(s.children, s.funs, num)

SeriesModel(x::Vararg{Any, N}) where N = SeriesModel(tuple(x)...)
Base.length(x::SeriesModel) = length(x.children)
Base.getindex(x::SeriesModel, i) = x.children[i]
function Base.iterate(c::SeriesModel, state = 0)
    state >= nfields(c) && return
    return Base.getfield(c, state+1), state+1
end


function CompositeModel(composite::SeriesModel) 
    Nstrain = number_strain_rate_components(composite)
    Nstress = number_stress_components(composite)
    #Num     = number_elements(composite)
    #new{Nstrain, Nstress, T, typeof(Num)}(composite, Num)

    update_global_numbers(s)        # update numbering

    return CompositeModel{Nstrain, Nstress, typeof(composite)}(composite)
end

struct ParallelModel{N, T, F} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    siblings::T # horizontal branching
    funs::F
    num::MVector{N, Int}
end
function ParallelModel(composite::T) where T
    funs = get_unique_state_functions(composite, :parallel)
    funs_flat = flatten_repeated_functions(funs)
    N = length(composite)
    ParallelModel{N, T, typeof(funs_flat)}(composite, funs_flat, MVector{N,Int}(1:N))
end
#update_numbers(s::ParallelModel{N, T, F}, num::NTuple) where {N, T, F} = ParallelModel{N, T, F}(s.siblings, s.funs, num)

ParallelModel(x::Vararg{Any, N}) where N = ParallelModel(tuple(x)...)
Base.length(x::ParallelModel) = length(x.siblings)
Base.getindex(x::ParallelModel, i) = x.siblings[i]
function Base.iterate(c::ParallelModel, state = 0)
    state >= nfields(c) && return
    return Base.getfield(c, state+1), state+1
end

number_strain_rate_components(::T) where T = 0
number_strain_rate_components(::CompositeModel{NStrain, NStress}) where {NStrain, NStress} = NStrain

@generated function number_strain_rate_components(x::ParallelModel{N0}) where N0
    quote
        (; siblings) = x
        N = 0
        Base.@nexprs $N0 i -> N += number_strain_rate_components(siblings[i])
        N
    end
end

@generated function number_strain_rate_components(x::SeriesModel{N0}) where N0
    quote
        (; children) = x
        N = N0
        Base.@nexprs $N0 i -> N += number_strain_rate_components(children[i])
        N
    end
end

number_stress_components(::T) where T = 0
number_stress_components(::CompositeModel{NStrain, NStress}) where {NStrain, NStress} = NStress

@generated function number_stress_components(x::SeriesModel{N0}) where N0
    quote
        (; children) = x
        N = 0
        Base.@nexprs $N0 i -> N += number_stress_components(children[i])
        N
    end
end

@generated function number_stress_components(x::ParallelModel{N0}) where N0
    quote
        (; siblings) = x
        N = N0
        Base.@nexprs $N0 i -> N += number_stress_components(siblings[i])
        N
    end
end

# @generated function get_model_state_functions(x::Composite)
#     funs_flat = flatten_repeated_functions(funs)
# end

# Boris stuff (thus likely needs rewriting)

 function number_elements1(s::SeriesModel{N}; start::Int64=1)  where N
     # this allocates but only needs to be done once...
     number = ()
     n = start-1;
     for i=1:N
         n += 1
         if isa(s.children[i], ParallelModel) 
         ##    # recursively deal with series (or parallel) elements
             numel  = number_elements1(s.children[i], start=n+1)
             number = (number..., (n,numel))
             n = maximum(maximum.(last.(numel)))
        elseif isa(s.children[i], SeriesModel) 
            numel  = number_elements1(s.children[i], start=n)
            number = (number..., (numel...,))
            n = maximum(maximum.(last.(numel)))
         else
             number = (number..., n)
         end
     end
   
     return number
 end

function number_elements1(s::ParallelModel{N}; start::Int64=1)  where N
    # this allocates but only needs to be done once...
    number = ()
    n = start-1;
    for i=1:N
        n += 1
        if isa(s.siblings[i], SeriesModel) 
            # recursively deal with series (or parallel) elements
            numel = number_elements1(s.siblings[i], start=n+1)
            number = (number..., (n,numel))
            n = maximum(maximum.(last.(numel)))
        elseif isa(s.siblings[i], SeriesModel) 
            numel  = number_elements1(s.siblings[i], start=n)
            number = (number..., (numel...,))
            n = maximum(maximum.(last.(numel)))
        else
            number = (number..., n)
        end
    end
  
    return number
end
number_elements1(c::CompositeModel) = number_elements1(c.components, start=1)

global_numbering(::NTuple{N, Int}) where N = ntuple(i -> i, Val(N))

function global_numbering(local_number::Tuple) 
    shift = Ref(0)
    ntuple(Val(length(local_number))) do i 
        @inline
        ntuple(Val(length(local_number[i]))) do j
            @inline
            shift[] += 1
            shift[]
        end
    end
end

function clean_numbers(numbers::NTuple{N, Any}) where N
    ntuple(Val(N)) do i
        @inline
        _clean_numbers(numbers[i])
    end
end

_clean_numbers(numbers::Int)                    = numbers
_clean_numbers(numbers::NTuple{1, Int})         = first(numbers)
_clean_numbers(numbers::NTuple{N, Int}) where N = numbers

function number_elements(c::CompositeModel)
    # local number of the components within a single Series/Parallel model
    local_number = number_elements(c.components, 0)
    # global numbering
    global_number = global_numbering(local_number)
    clean_numbers(global_number)
end

@inline number_elements(::AbstractRheology, ::Any) = 1

@generated function number_elements(s::SeriesModel{N}, ::C) where {N,C} 
    quote 
        @inline
        Base.@ntuple $N i -> number_elements(s.children[i],i)
    end
end

@generated function number_elements(s::ParallelModel{N}, ::C) where {N,C} 
    quote 
        @inline
        Base.@ntuple $N i -> number_elements(s.siblings[i], i)
    end
end


@inline parallel_state_functions(r::ParallelModel) = series_state_functions(r.siblings)
@inline parallel_state_functions(::SeriesModel) = (compute_strain_rate, compute_stress,)
@inline series_state_functions(::ParallelModel) = (compute_strain_rate, compute_stress,)
@inline series_state_functions(r::SeriesModel) = series_state_functions(r.children, r.num)


# recursively updates the numbers of the elements
function update_global_numbers(s::Union{SeriesModel{N},ParallelModel{N}}, start=0) where N
    s.num .= s.num .+ start
    start = maximum(s.num)
    for i=1:N
       start = update_global_numbers(s[i], start)
    end
    return start
end

function update_global_numbers(s::AbstractRheology, start=0)
    return start
end