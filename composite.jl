abstract type AbstractCompositeModel <: AbstractRheology  end

@inline series_state_functions(::AbstractCompositeModel)= ()
@inline parallel_state_functions(::AbstractCompositeModel)= ()

struct CompositeModel{Nstrain, Nstress, T} <: AbstractCompositeModel
    components::T

    function CompositeModel(composite::T) where T
        Nstrain = number_strain_rate_components(composite)
        Nstress = number_stress_components(composite)
        new{Nstrain, Nstress, T}(composite)
    end
end
CompositeModel(x::Vararg{Any, N}) where N = CompositeModel(tuple(x)...)

struct SeriesModel{N, T, F} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    children::T # vertical stacking
    funs::F

    function SeriesModel(composite::T) where T
        funs = get_unique_state_functions(composite, :series)
        funs_flat = flatten_repeated_functions(funs)
        N = length(composite)
        new{N, T, typeof(funs_flat)}(composite, funs_flat)
    end
end
SeriesModel(x::Vararg{Any, N}) where N = SeriesModel(tuple(x)...)

struct ParallelModel{N, T, F} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    siblings::T # horizontal branching
    funs::F
    function ParallelModel(composite::T) where T
        funs = get_unique_state_functions(composite, :parallel)
        funs_flat = flatten_repeated_functions(funs)
        N = length(composite)
        new{N, T, typeof(funs_flat)}(composite, funs_flat)
    end
end
ParallelModel(x::Vararg{Any, N}) where N = ParallelModel(tuple(x)...)

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