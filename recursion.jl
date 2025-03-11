using Test
using DispatchDoctor: @stable
using StaticArrays

import Base.IteratorsMD.flatten
abstract type AbstractRheology end
abstract type AbstractPlasticity <: AbstractRheology end # in case we need spacilization at some point

abstract type AbstractCompositeModel  end

include("src/rheology_types.jl")
include("src/state_functions.jl")
include("src/kwargs.jl")
# include("../src/matrices.jl")
include("albert/others.jl")
include("albert/residual.jl")
include("albert/composite.jl")

################ BASIC STRUCTS

struct CompositeModel{Nstrain, Nstress, T} <: AbstractCompositeModel
    components::T
end

struct SeriesModel{L, B} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    leafs::L     # horizontal stacking
    branches::B  # vertical stacking

    function SeriesModel(c::Vararg{Any, N}) where N
        leafs = series_leafs(c)
        branches = series_branches(c)
        new{typeof(leafs), typeof(branches)}(leafs, branches)
    end
end

Base.show(io::IO, ::SeriesModel) = print(io, "SeriesModel")

struct ParallelModel{L, B} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    leafs::L     # horizontal stacking
    branches::B  # vertical stacking

    function ParallelModel(c::Vararg{Any, N}) where N
        leafs    = parallel_leafs(c)
        branches = parallel_branches(c)
        new{typeof(leafs), typeof(branches)}(leafs, branches)
    end
end

Base.show(io::IO, ::ParallelModel) = print(io, "ParallelModel")

@inline series_leafs(c::NTuple{N, AbstractRheology}) where N = c
@inline series_leafs(c::AbstractRheology) = (c,)
@inline series_leafs(::ParallelModel) = ()
@inline series_leafs(::Tuple{}) = ()
@inline series_leafs(c::NTuple{N, Any}) where N = series_leafs(first(c))..., series_leafs(Base.tail(c))...

@inline parallel_leafs(c::NTuple{N, AbstractRheology}) where N = c
@inline parallel_leafs(c::AbstractRheology) = (c,)
@inline parallel_leafs(::SeriesModel) = ()
@inline parallel_leafs(::Tuple{}) = ()
@inline parallel_leafs(c::NTuple{N, Any}) where N = parallel_leafs(first(c))..., parallel_leafs(Base.tail(c))...

@inline series_branches(::NTuple{N, AbstractRheology}) where N = ()
@inline series_branches(::AbstractRheology) = ()
@inline series_branches(c::ParallelModel) = (c,)
@inline series_branches(::Tuple{}) = ()
@inline series_branches(c::NTuple{N, Any}) where N = series_branches(first(c))..., series_branches(Base.tail(c))...

@inline parallel_branches(::NTuple{N, AbstractRheology}) where N = ()
@inline parallel_branches(::AbstractRheology) = ()
@inline parallel_branches(c::SeriesModel) = (c,)
@inline parallel_branches(::Tuple{}) = ()
@inline parallel_branches(c::NTuple{N, Any}) where N = parallel_branches(first(c))..., parallel_branches(Base.tail(c))...
################ START RECURSIVE FUNCTIONS AND TESTING

"""
    parallel_numbering(c::SeriesModel; counter::Base.RefValue{Int64} = Ref(0))

Assigns a unique number to each parallel element within a `SeriesModel` object `c`. 
The numbering starts from the value of `counter` and increments for each element.

# Arguments
- `c::SeriesModel`: The `SeriesModel` object whose elements are to be numbered.
- `counter::Base.RefValue{Int64}`: A reference to an integer that keeps track of the current number. 
  Defaults to `Ref(0)`.
"""
@stable function parallel_numbering(c::Union{ParallelModel,SeriesModel}; counter::Base.RefValue{Int64} = Ref(0))
    (; branches) = c

    np = length(branches)

    # NOTE: counter[] is mutated "globally" within the recursion stack
    numbering = ntuple(Val(np)) do j
        @inline 
        c0 = counter[] += 1
        inner_branches = branches[j].branches
        x = ntuple(Val(length(inner_branches))) do i 
            @inline 
            parallel_numbering(inner_branches[i]; counter = counter)
        end 
        c0, x...
    end
    numbering
end

#@inline @stable parallel_numbering(::Union{Tuple{}, ParallelModel}; counter::Base.RefValue{Int64} = Ref(0)) = ()
@inline @stable parallel_numbering(::Tuple{}; counter::Base.RefValue{Int64} = Ref(0)) = ()


@stable function series_numbering(c::ParallelModel; counter::Base.RefValue{Int64} = Ref(0))
    (; branches) = c

    np = length(branches)

    numbering = ntuple(Val(np)) do j
        @inline 
        c0 = counter[] += 1
        inner_branches = branches[j].branches
        x = ntuple(Val(length(inner_branches))) do i 
            @inline 
            s = series_numbering(inner_branches[i]; counter = counter)
            counter[] += 1
            s
        end 
        c0, x...
    end
    numbering
end

@stable function series_numbering(c::SeriesModel; counter::Base.RefValue{Int64} = Ref(0))
    (; branches) = c

    np = length(branches)

    numbering = ntuple(Val(np)) do j
        @inline 
        c0 = counter[] += 1
        inner_branches = branches[j].branches
        x = ntuple(Val(length(inner_branches))) do i 
            @inline 
            s = series_numbering(inner_branches[i]; counter = counter)
            counter[] += 1
            s
        end 
        (c0,), x...
    end
    numbering
end

@inline @stable series_numbering(::Tuple{};     counter::Base.RefValue{Int64} = Ref(0)) = ()
#@inline @stable series_numbering(::SeriesModel; counter::Base.RefValue{Int64} = Ref(0)) = counter


# testing grounds

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)

composite  = viscous1, powerlaw
c1 = let
    s1 = SeriesModel(viscous1, viscous2)
    s2 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, drucker, p)
end

c2 = let
    p1  = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

c3 = let
    p1  = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

c3 = let
    p1  = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p, p)
end

c4 = let
    p1  = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, p1)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

@test parallel_numbering(c1) == ((1, ()),)
@test parallel_numbering(c2) == ((1, ((2,),)),)
@test parallel_numbering(c3) == ((1, ((2,),)), (3, ((4,),)))
@test parallel_numbering(c4) == ((1, ((2,), (3,))),)