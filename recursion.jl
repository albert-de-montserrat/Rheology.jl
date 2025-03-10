using Test
using DispatchDoctor: @stable

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
@stable function parallel_numbering(c::SeriesModel; counter::Base.RefValue{Int64} = Ref(0))
    (; branches) = c

    np = length(branches)

    numbering = ntuple(Val(np)) do j
        @inline 
        c0 = counter[] += 1
        inner_branches = branches[j].branches
        x = ntuple(Val(length(inner_branches))) do i 
            @inline 
            s = parallel_numbering(inner_branches[i]; counter = counter)
            counter[] += 1
            s
        end 
        c0, x...
    end
    numbering
end

@inline @stable parallel_numbering(::Union{Tuple{}, ParallelModel}; counter::Base.RefValue{Int64} = Ref(0)) = ()

# testing grounds

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)

composite  = viscous1, powerlaw
c1 = let
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
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

@test parallel_numbering(c1) == ((1, ()),)
@test parallel_numbering(c2) == ((1, ((2,),)),)
