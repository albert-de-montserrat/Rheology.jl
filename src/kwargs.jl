@inline function augment_args(args, Δx)
    k = keys(args)
    vals = MVector(values(args))
    for i in eachindex(Δx)
        vals[i] += Δx[i]
    end
    return (; zip(k, vals)...)
end

@inline function update_args2(args, x::SVector{N, T}) where {N,T}
    k = keys(args)
    N0 = length(args)
    vals = @MVector zeros(T, N0)
    for i in 1:length(args)
        vals[i] = x[i]
    end
    return (; zip(k, vals)...)
end

# dummy NamedTuple allocators

@inline residual_kwargs(::Type{T}, ::Function)                                       where T = (; tmp = zero(T))
@inline residual_kwargs(::Type{T}, ::typeof(compute_strain_rate))                    where T = (; ε = zero(T),)
@inline residual_kwargs(::Type{T}, ::typeof(compute_volumetric_strain_rate))         where T = (; θ = zero(T))
@inline residual_kwargs(::Type{T}, ::typeof(compute_stress))                         where T = (; τ = zero(T),)
@inline residual_kwargs(::Type{T}, ::typeof(compute_pressure))                       where T = (; P = zero(T),)
# @inline residual_kwargs(::Type{T}, ::typeof(compute_lambda))                         where T = (; λ = zero(T)) # τ = zero(T), P = zero(T))
# @inline residual_kwargs(::Type{T}, ::typeof(compute_plastic_strain_rate))            where T = (; τ_pl = zero(T),)
# @inline residual_kwargs(::Type{T}, ::typeof(compute_plastic_stress))                 where T = (; τ_pl = zero(T),)
# @inline residual_kwargs(::Type{T}, ::typeof(compute_volumetric_plastic_strain_rate)) where T = (; τ_pl = zero(T), P_pl = zero(T))

@inline residual_kwargs(funs::F)              where F<:Function = residual_kwargs(Float64, funs)
@inline residual_kwargs(funs::NTuple{N, Any}) where N           = residual_kwargs.(Float64, funs)

# dummy NamedTuple allocators
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_strain_rate))                    where T = (; τ = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_volumetric_strain_rate))         where T = (; P = zero(T))
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_lambda))                         where T = (; λ = zero(T)) # τ = zero(T), P = zero(T))
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_stress))                         where T = (; ε = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_pressure))                       where T = (; θ = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_plastic_strain_rate))            where T = (; τ_pl = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_plastic_stress))                 where T = (; τ_pl = zero(T),)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_volumetric_plastic_strain_rate)) where T = (; τ_pl = zero(T), P_pl = zero(T))
@inline differentiable_kwargs(::Type{T}, ::typeof(state_var_reduction))                    where T = (; )

# add numbers to the differentiable_kwargs as long as they are not part of the standard series variables
function attach_nums(x::NamedTuple, n::Int64)
    # This allocates - to be fixed!
    k = keys(x)
    if n>0
        k_new = string.(keys(x)).*"_$n"
        N = length(x)
        k1 = ntuple(i-> Symbol(k_new[i]) , Val(N))
    else
        k1 = k
    end
    return NamedTuple{ k1}(values(x))
end

@inline differentiable_kwargs(::Type{T}, ::typeof(compute_strain_rate), i::Int64)                    where T = attach_nums((; τ = zero(T),), i)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_volumetric_strain_rate), i::Int64)         where T = attach_nums((; τ = zero(T), P = zero(T)),i)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_lambda), i::Int64)                         where T = attach_nums((; λ = zero(T)),i) # τ = zero(T), P = zero(T))
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_stress), i::Int64)                         where T = attach_nums((; ε = zero(T),),i)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_pressure), i::Int64)                       where T = attach_nums((; θ = zero(T),),i)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_plastic_strain_rate), i::Int64)            where T = attach_nums((; τ_pl = zero(T),),i)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_plastic_stress), i::Int64)                 where T = attach_nums((; τ_pl = zero(T),),i)
@inline differentiable_kwargs(::Type{T}, ::typeof(compute_volumetric_plastic_strain_rate), i::Int64) where T = attach_nums((; τ_pl = zero(T), P_pl = zero(T)),i)

@inline differentiable_kwargs(funs::NTuple{N, Any}) where N = differentiable_kwargs(Float64, funs)
@inline differentiable_kwargs(funs::NTuple{N, Any}, nums::NTuple{N,Any}) where N = differentiable_kwargs(Float64, funs, nums)

@generated function differentiable_kwargs(::Type{T}, funs::NTuple{N, Any}) where {N, T}
    quote
        @inline 
        Base.@nexprs $N i -> nt_i = differentiable_kwargs($T, funs[i])
        Base.@ncall $N merge nt
    end
end

@generated function differentiable_kwargs(::Type{T}, funs::NTuple{N, Any}, nums::NTuple{N,I}) where {N, T, I}
    quote
        @inline 
        Base.@nexprs $N i -> nt_i = differentiable_kwargs($T, funs[i], nums[i])
        Base.@ncall $N merge nt
    end
end


@inline all_differentiable_kwargs(funs::NTuple{N, Any}) where N = all_differentiable_kwargs(Float64, funs)

@generated function all_differentiable_kwargs(::Type{T}, funs::NTuple{N, Any}) where {N, T}
    quote
        @inline 
        Base.@ntuple $N i -> differentiable_kwargs($T, funs[i])
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
            numel = number_elements(composite[i].elements, start=n+1)
            number = (number..., (n,numel))
            n = maximum(numel)
        else
            number = (number..., n)
        end
    end
    
    return number
end


