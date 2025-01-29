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
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_lambda))                         where T = (; τ = zero(T), P = zero(T))
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_stress))                         where T = (; ε = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_pressure))                       where T = (; θ = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_plastic_strain_rate))            where T = (; τ_pl = zero(T),)
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
    args_diff = Base.structdiff(args, args_nondiff)
    
    return args_diff, args_nondiff
end