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


@inline differentiable_kwargs(::typeof(compute_shear_strain)) = (:τ,)
@inline differentiable_kwargs(::typeof(compute_volumetric_strain)) = :τ, :P
@inline differentiable_kwargs(::typeof(compute_lambda)) = :τ, :P
@inline differentiable_kwargs(::typeof(compute_stress)) = (:ε,)
@inline differentiable_kwargs(::typeof(compute_pressure)) = (:θ,)

differentiable_kwargs(funs::NTuple{N, Any}) where N = flatten_repeated_functions(Base.IteratorsMD.flatten(differentiable_kwargs.(funs)))

length_difference(::NTuple{N1}, ::NTuple{N2}) where {N1, N2} = Val(N1-N2)

nondifferentiable_kwargs(v::NTuple{N}, k::NTuple{N}, dk::NTuple{N}, ::Val{0}) where {N} = (; zip(k, v)...)

# @generated function nondifferentiable_kwargs(v::NTuple{N1, Number}, k::NTuple{N1, Symbol}, dk::NTuple{N2, Symbol}) where {N1, N2}
#     quote
#         @inline 
#         # not_dk = Base.@ntuple $N1 i -> (k[i] ∉ dk ? (k[i],) : ())
#         # Base.IteratorsMD.flatten(not_dk)

#         Base.Base.@nexprs $N2 i -> begin
#             Base.Base.@nexprs $N1 j -> begin
#         (k[i] ∉ dk ? (k[i],) : ())
#         Base.Base.@nexprs $N1 i -> (k[i] ∉ dk ? (k[i],) : ())

#     end
# end


@generated function nondifferentiable_kwargs_inds(v::NTuple{N1}, k::NTuple{N1}, dk::NTuple{N2}, ::Val{ND}) where {N1, N2, ND}
    f(k, dk, ::Val{N}) where N = MVector{N}(k[i] ∈ dk for i in 1:N)
    f(::Val{N})        where N = MVector{N}(0 for i in 1:N)
    
    quote
        @inline 
        flags = f(k, dk, Val($N1))
        inds = f(Val($ND))
        Base.@nexprs $ND i -> begin
            Base.@nexprs $N1 j -> begin
                if flags[j] == false && !(dk[i] === k[j])
                    inds[i] = j
                    flags[j] = true
                end
            end
        end
        # tuple(inds...)

        kk = Base.@ntuple $ND i -> k[inds[i]]
        vv = Base.@ntuple $ND i -> v[inds[i]]
        # kk, vv
        NamedTuple{kk}(vv)
        # pk=:potato
        # pv=1
        # (; zip(kk, vv)...)
    end
end

@code_warntype nondifferentiable_kwargs_inds(v, k, dk, l_diff)
@b nondifferentiable_kwargs_inds($(v, k, dk, l_diff)...)
inds = nondifferentiable_kwargs_inds(v, k, dk, l_diff)

function bar(k::NTuple, v::NTuple, inds::NTuple{N}) where N
    kk = ntuple(Val(N)) do i
        @inline
        k[inds[i]]
    end
    vv = ntuple(Val(N)) do i
        @inline
        v[inds[i]]
    end
    kk,vv
    # (; zip(kk, vv)...)
end

@code_warntype bar(k, v, inds)
@b bar($(k, v, inds)...)

kk, vv = bar(k, v, inds)
h(kk, vv) = (; zip(kk, vv)...)
@code_warntype h((:a,), (2,))

ntuple(Val(length(inds))) do i
    k[inds[i]]
end

@code_warntype ntuple(Val(1)) do i
    inds[i]
end

@b nondifferentiable_kwargs($(v, k, dk, l_diff)...)


@code_warntype nondifferentiable_kwargs(v, k, dk)
@code_warntype nondifferentiable_kwargs(k, dk)
@b nondifferentiable_kwargs($(v, k, dk)...)
@b nondifferentiable_kwargs($(k, dk)...)

@b differentiable_kwargs($statefuns)

dk     = differentiable_kwargs(statefuns)
k      = keys(args)
v      = values(args)
non_dk = nondifferentiable_kwargs(v, k, dk)
l_diff = length_difference(k, dk)

nondifferentiable_kwargs(v, k, dk, l_diff)

function foo(non_dk::NTuple{N, Symbol}, args) where N
    ntuple(Val(N)) do i
        @inline
        getfield(args, non_dk[i])
    end
end

@b foo($(non_dk, args)...)