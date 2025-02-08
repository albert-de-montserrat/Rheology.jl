@inline function augment_args(args, Δx)
    k = keys(args)
    vals = MVector(values(args))
    for i in eachindex(Δx)
        vals[i] += Δx[i]
    end
    return (; zip(k, vals)...)
end

@inline function update_args(args, Δx)
    k = keys(args)
    vals = MVector(values(args))
    for i in eachindex(Δx)
        vals[i] = Δx[i]
    end
    return (; zip(k, vals)...)
end

# dummy NamedTuple allocators
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_strain_rate))                    where T = (; τ = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_volumetric_strain_rate))         where T = (; τ = zero(T), P = zero(T))
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_lambda))                         where T = (; λ = zero(T)) # τ = zero(T), P = zero(T))
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_stress))                         where T = (; ε = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_pressure))                       where T = (; θ = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_plastic_strain_rate))            where T = (; τ_pl = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_plastic_stress))                 where T = (; τ_pl = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_volumetric_plastic_strain_rate)) where T = (; τ_pl = zero(T), P_pl = zero(T))

differentiable_kwargs(funs::NTuple{N, Any}) where N = differentiable_kwargs(Float64, funs)

@generated function differentiable_kwargs(::Type{T}, funs::NTuple{N, Any}) where {N, T}
    quote
        @inline 
        Base.@nexprs $N i -> nt_i = differentiable_kwargs($T, funs[i])
        Base.@ncall $N merge nt
    end
end

function split_args(args, statefuns::NTuple{N, Any}) where N
    # split args into differentiable and not differentiable
    dummy = differentiable_kwargs(statefuns)
    args_nondiff = Base.structdiff(args, dummy)
    args_diff  = Base.structdiff(dummy, args_nondiff)
    args_diff0 = Base.structdiff(args, args_nondiff)
    
    args_diff = merge(args_diff, args_diff0)
    return args_diff, args_nondiff
end

"""
    number_elements(composite::NTuple{N, AbstractRheology}; start::Int64=1)  where N

Recursively creates a unique number for each rheological element
"""
function number_elements(composite::NTuple{N, AbstractRheology}; start::Int64=1)  where N
    # this allocates but only needs to be done once...
    number = ()
    n = start-1;
    for i=1:N
        n += 1
        if isa(composite[i], Series) || isa(composite[i], Parallel)
            # recursively deal with series (or parallel) elements
            numel = number_elements(composite[i].elements, start=n)
            @show numel
            number = (number..., numel)
            n = maximum(numel)
        else
            number = (number..., n)
        end

    end
    
    return number
end