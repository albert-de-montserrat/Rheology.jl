abstract type AbstractRheology end
abstract type AbstractPlasticity <: AbstractRheology end # in case we need spacilization at some point

abstract type AbstractCompositeModel <: AbstractRheology  end

@inline series_state_functions(::AbstractCompositeModel)= ()
@inline parallel_state_functions(::AbstractCompositeModel)= ()

struct CompositeModel{Nstrain, Nstress, T} <: AbstractCompositeModel
    components::T
end
function CompositeModel(composite::T) where T
    Nstrain = number_strain_rate_components(composite)
    Nstress = number_stress_components(composite)
    return CompositeModel{Nstrain, Nstress, T}(composite)
end
CompositeModel(x::Vararg{Any, N}) where N = CompositeModel(tuple(x)...)

struct SeriesModel{N, T, F} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    children::T # vertical stacking
    funs::F
    num::MVector{N, Int}
    n::MVector{1, Int}       # number of this ParallelModel
    parent::MVector{1, Int}  # number of the parent Series/Parallel element
end


function SeriesModel(composite::T) where T
    funs = get_unique_state_functions(composite, :series)
    funs_flat = flatten_repeated_functions(funs)
    N = length(composite)
    SeriesModel{N, T, typeof(funs_flat)}(composite, funs_flat,  MVector{N,Int}(1:N), MVector{1,Int}(0), MVector{1,Int}(0))
end
#update_numbers(s::SeriesModel{N, T, F}, num::NTuple) where {N, T, F} = SeriesModel{N, T, F}(s.children, s.funs, num)
isseries(x::AbstractRheology) = false
isseries(x::SeriesModel) = true

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
    num::MVector{N, Int}    # numbers of the rheological elements
    n::MVector{1, Int}      # number of this ParallelModel
    parent::MVector{1, Int}  # number of the parent Series/Parallel element
end
function ParallelModel(composite::T) where T
    funs = get_unique_state_functions(composite, :parallel)
    funs_flat = flatten_repeated_functions(funs)
    N = length(composite)
    ParallelModel{N, T, typeof(funs_flat)}(composite, funs_flat, MVector{N,Int}(1:N), MVector{1,Int}(0), MVector{1,Int}(0))
end

ParallelModel(x::Vararg{Any, N}) where N = ParallelModel(tuple(x)...)
Base.length(x::ParallelModel) = length(x.siblings)
Base.getindex(x::ParallelModel, i) = x.siblings[i]
function Base.iterate(c::ParallelModel, state = 0)
    state >= nfields(c) && return
    return Base.getfield(c, state+1), state+1
end

isparallel(x::AbstractRheology) = false
isparallel(x::ParallelModel) = true


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


@inline parallel_state_functions(r::ParallelModel) = parallel_state_functions(r.siblings, r.num)
@inline parallel_state_functions(::SeriesModel) = (compute_stress,)
@inline series_state_functions(::ParallelModel) = (compute_stress, )
@inline series_state_functions(r::SeriesModel) = series_state_functions(r.children, r.num)


# recursively updates the numbers of the elements
function update_global_numbers(s::Union{SeriesModel{N},ParallelModel{N}}, num=0, start=0, parent=0) where N
    s.num       .= s.num .+ start
    s.n[1]       = num
    s.parent[1]  = parent
    parent       = start
    start = maximum(s.num)
    for i=1:N
       #s[i].n[1] = i
       start, _ = update_global_numbers(s[i], s.num[i], start, parent)
    end
   
    return start, parent
end

function update_global_numbers(s::AbstractRheology, num=0, start=0, parent=0)
    return start, parent
end


###
# deal with kwargs


# Switch the value in a tuple
function switch_k(tup::NTuple{N,T}, elem::T, pos::Integer) where {N,T}
    return ntuple(i-> (pos!==i) ? tup[i] : elem , Val(N))
end

update_statenums(r::AbstractRheology, num::Int, i::Int, statenums) = statenums
update_statenums(r::ParallelModel, num::Int, i::Int, statenums) = switch_k(statenums, num, i)

# retrieve keywords of differentiable variable and of variables needed for the rhs  
function get_kwargs(s::SeriesModel{N,T}; level_up=nothing) where {N,T}

    statefuns, statenums, stateelements = series_state_functions(s)
    if !isnothing(level_up)
        num_model = level_up
    else
        num_model = s.n[1]
    end

    parallel    = isparallel.(s.children)
    series      = isseries.(s.children)

    # the ones that are not parallel or series will be merged if they 
    # are strain rate related (for a SeriesModel)
    for (i,funs) in enumerate(statefuns)
        if (funs == compute_strain_rate || funs == compute_volumetric_strain_rate) && 
            !parallel[stateelements[i]] && !series[stateelements[i]]
            
            statenums = switch_k(statenums, num_model, i)
        end
    end
   
    num_model =  s.n[1]
    # get NamedTuples with underscore and number
    # THIS ALLOCATES because attach_nums allocates -> to be fixed
    diff_args = differentiable_kwargs(statefuns, statenums)
   
    # attach the values of the full element here
    if num_model>0
        ε_series = NamedTuple{(Symbol("ε_$num_model"),)}(zero(eltype(diff_args)))
        θ_series = NamedTuple{(Symbol("θ_$num_model"),)}(zero(eltype(diff_args)))
    else
        ε_series = NamedTuple{(:ε,)}(zero(eltype(diff_args)))
        θ_series = NamedTuple{(:θ,)}(zero(eltype(diff_args)))
    end
    res_args = ε_series
    if isvolumetric(s)
        res_args = merge(θ_series, ε_series)
    end

    return diff_args, res_args
end

function get_kwargs(p::ParallelModel{N,T}; level_up=nothing) where {N,T}

    statefuns, statenums, stateelements = parallel_state_functions(p)
    if !isnothing(level_up)
        num_model = level_up
    else
        num_model = p.n[1]
    end

    parallel    = isparallel.(p.siblings)
    series      = isseries.(p.siblings)

    # the ones that are not parallel or series will be merged if they 
    # are strain rate related (for a SeriesModel)
    for (i,funs) in enumerate(statefuns)
        if (funs == compute_stress || funs == compute_pressure) && 
            !parallel[stateelements[i]] && !series[stateelements[i]]
            
            statenums = switch_k(statenums, num_model, i)
        end
    end
   
    num_model =  s.n[1]

    # get NamedTuples with underscore and number
    # THIS ALLOCATES because attach_nums allocates -> to be fixed
    diff_args = differentiable_kwargs(statefuns, statenums)
   
    # attach the values of the full element here
    if num_model>0
        τ_series = NamedTuple{(Symbol("τ_$num_model"),)}(zero(eltype(diff_args)))
        P_series = NamedTuple{(Symbol("P_$num_model"),)}(zero(eltype(diff_args)))
    else
        τ_series = NamedTuple{(:τ,)}(zero(eltype(diff_args)))
        P_series = NamedTuple{(:P,)}(zero(eltype(diff_args)))
    end
    res_args = τ_series
    if isvolumetric(s)
        res_args = merge(P_series, τ_series)
    end

    return diff_args, res_args
end



get_all_kwargs(s::Union{SeriesModel,ParallelModel}) = get_all_kwargs(s, get_kwargs(s)...)

function get_all_kwargs(s::Union{SeriesModel{N,T},ParallelModel{N,T}}, diff_args, res_args) where {N,T}
    for i=1:N
        diff_args1, res_args1 = get_kwargs(s[i])
        diff_args = merge(diff_args, diff_args1)
        res_args  = merge(res_args, res_args1)
    end

    return diff_args, res_args
end

get_all_kwargs(s::AbstractRheology, diff_args, res_args) = NamedTuple(), NamedTuple()
get_kwargs(s::AbstractRheology; level_up=nothing) = NamedTuple(), NamedTuple()

get_all_kwargs(c::CompositeModel) = get_all_kwargs(c.components)