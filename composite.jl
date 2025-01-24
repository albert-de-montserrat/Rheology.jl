abstract type AbstractCompositeModel  end

struct CompositeModel{T} <: AbstractCompositeModel
    components::T
end
CompositeModel(x::Vararg{Any, N}) where N = CompositeModel(tuple(x)...)

struct SeriesModel{T} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    children::T # vertical stacking
end
SeriesModel(x::Vararg{Any, N}) where N = SeriesModel(tuple(x)...)

struct ParallelModel{T} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    siblings::T # horizontal branching
end
ParallelModel(x::Vararg{Any, N}) where N = ParallelModel(tuple(x)...)

# # example or parallel-series model
# p = ParallelModel(viscous, drucker)
# s = SeriesModel(elastic, powerlaw, p)
# composite = CompositeModel(s, p)
