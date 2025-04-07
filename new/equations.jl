include("recursion.jl")

struct CompositeEquation{IsGlobal, T, F, R}
    parent::Int64  # i-th element of x to be substracted
    child::T       # i-th element of x to be added
    self::Int64    # equation number
    fn::F          # state function
    rheology::R 
    ind_input::Int64

    function CompositeEquation(parent::Int64, child::T, self::Int64, fn::F, rheology::R, ind_input, ::Val{B}) where {T, F, R, B}
        @assert B isa Bool
        new{B, T, F, R}(parent, child, self, fn, rheology, ind_input)
    end
end


# struct CompositeEquation{IsGlobal, T, F, R}
#     parent::Int64  # i-th element of x to be substracted
#     child::T       # i-th element of x to be added
#     self::Int64    # equation number
#     fn::F          # state function
#     rheology::R 

#     function CompositeEquation(parent::Int64, child::T, self::Int64, fn::F, rheology::R, ::Val{B}) where {T, F, R, B}
#         @assert B isa Bool
#         new{B, T, F, R}(parent, child, self, fn, rheology)
#     end
# end

# function generate_equations(c::AbstractCompositeModel; iparent::Int64 = 0, iself::Int64 = 0)
#     iself_ref = Ref{Int64}(iself)

#     (; branches, leafs) = c
    
#     fns_own_global, fns_own_local = get_own_functions(c)
#     fns_branches_global,          = get_own_functions(branches)

#     nown             = length(fns_own_global)
#     nlocal           = length(fns_own_local)
#     nbranches        = length(branches)

#     ilocal_childs    = ntuple(i -> iparent + nown - 1 + i, Val(nlocal))
#     offsets_parallel = (0, length.(fns_branches_global)...)
#     iparallel_childs = ntuple(i -> iparent + nlocal + offsets_parallel[i] + i + nown, Val(nbranches))

#     # add global equations
#     global_eqs = add_global_equations(iparent, iparallel_childs, iself_ref, fns_own_global, leafs, branches, Val(nown))
#     # add local equations
#     local_eqs =  add_local_equations(iparent, ilocal_childs, iself_ref, fns_own_local, leafs, Val(nlocal))
#     # add parallel equations
#     parallel_eqs = add_parallel_equations(global_eqs, branches, iself_ref, fns_own_global)
    
#     return (global_eqs..., local_eqs..., parallel_eqs...) |> superflatten 
# end
@inline generate_equations(c::AbstractCompositeModel) = generate_equations(c, global_series_functions(c))

@generated function generate_equations(c::AbstractCompositeModel, fns::NTuple{N, Any}) where N
    quote
        iparent = 0
        iself = 0
        isGlobal = Val(true)
        eqs = Base.@ntuple $N i -> begin
            @inline
            ind_input = i
            eqs = generate_equations(c, fns[i], ind_input, isGlobal, isvolumetric(c); iparent = iparent, iself = iself)
            iself = eqs[end].self 
            iparent = 0
            eqs
        end
        superflatten(eqs)
    end
end

function generate_equations(c::AbstractCompositeModel, fns_own_global::F, ind_input, ::Val{B}, ::Val; iparent::Int64 = 0, iself::Int64 = 0) where {F, B}
    iself_ref = Ref{Int64}(iself)
    (; branches, leafs) = c
    
    _, fns_own_local      = get_own_functions(c)
    # fns_branches_global,_ = get_own_functions(branches)

    nown             = 1 # length(fns_own_global)
    nlocal           = length(fns_own_local)
    nbranches        = length(branches)

    # iglobal          = ntuple(i -> iparent + i - 1, Val(nown))
    ilocal_childs    = ntuple(i -> iself + nown - 1 + i, Val(nlocal))
    offsets_parallel = (0, ntuple(i -> i, Val(nbranches))...)
    # offsets_parallel = (0, length.(fns_branches_global)...)
    iparallel_childs = ntuple(i -> iself + nlocal + offsets_parallel[i] + i + nown, Val(nbranches))

    # add globals
    # iself_ref[] += 1
    # global_eqs   = CompositeEquation(iparent, iparallel_childs, iself_ref[], fns_own_global, leafs, Val(false))
    isGlobal     = Val(B)
    global_eqs   = add_global_equations(iparent, iparallel_childs, iself_ref, fns_own_global, leafs, branches, ind_input, isGlobal, Val(1))

    local_eqs    = add_local_equations(iparent, ilocal_childs, iself_ref, fns_own_local, leafs, Val(nlocal))
    
    iparent_new  = global_eqs.self
    fn           = counterpart(fns_own_global)
    parallel_eqs = ntuple(Val(nbranches)) do i
        @inline
        generate_equations(branches[i], fn, 0, Val(false), isvolumetric(branches[i]); iparent = iparent_new, iself = iself_ref[])
    end

    return (global_eqs, local_eqs..., parallel_eqs...) |> superflatten
end

# eliminate equations
for fn in (:compute_pressure, :compute_volumetric_strain_rate)
    @eval begin
        @inline generate_equations(::AbstractCompositeModel, ::typeof($fn), ::Integer, ::Val, ::Val{false}; kwargs...) = ()
    end
end
@inline generate_equations(::Tuple{}; kwargs...) = ()

#### 

fn_pairs = (
    (compute_strain_rate, compute_stress),
    (compute_volumetric_strain_rate, compute_pressure),
)

for pair in fn_pairs
    @eval begin
        counterpart(::typeof($(pair[1]))) = $(pair[2])
        counterpart(::typeof($(pair[2]))) = $(pair[1])
    end
end

get_own_functions(c::NTuple{N, AbstractCompositeModel}) where N = ntuple(i -> get_own_functions(c[i]), Val(N))
get_own_functions(c::SeriesModel)                               = get_own_functions(c, series_state_functions, global_series_state_functions, local_series_state_functions)
get_own_functions(c::ParallelModel)                             = get_own_functions(c, parallel_state_functions, global_parallel_state_functions, local_parallel_state_functions)

function get_own_functions(c::AbstractCompositeModel, fn_state::F1, fn_global::F2, fn_local::F3) where {F1, F2, F3}
    fns_own_all    = fn_state(c.leafs)
    fns_own_global = fn_global(fns_own_all) |> superflatten |> flatten_repeated_functions
    fns_own_local  = fn_local(fns_own_all)
    fns_own_global, fns_own_local
end

get_own_functions(::Tuple{}) = (), ()
# get_own_functions(::Tuple{}) = compute_strain_rate, ()

get_local_functions(c::NTuple{N, AbstractCompositeModel}) where N = ntuple(i -> get_own_functions(c[i]), Val(N))

function get_local_functions(c::SeriesModel)
    fns_own_all    = series_state_functions(c.leafs)
    local_series_state_functions(fns_own_all)
end

function get_local_functions(c::ParallelModel)
    fns_own_all    = parallel_state_functions(c.leafs)
    local_parallel_state_functions(fns_own_all)
end

@inline has_children(::F, branch)                              where F = Val(true)
@inline has_children(::typeof(compute_pressure), branch)               = isvolumetric(branch)
@inline has_children(::typeof(compute_volumetric_strain_rate), branch) = isvolumetric(branch)

@inline correct_children(fn::F, branch::AbstractCompositeModel, children) where F = correct_children(children, has_children(fn, branch))
@generated function correct_children(fn::F, branch::NTuple{N,AbstractCompositeModel}, children) where {F, N} 
    quote
        new_children = Base.@ntuple $N i -> correct_children(children[i], has_children(fn, branch[i]))
        return superflatten(new_children )
    end
end
@inline correct_children(children, ::Val{true})  = children
@inline correct_children(::Any,    ::Val{false}) = ()

@generated function add_global_equations(iparent, iparallel_childs, iself_ref, fns_own_global::NTuple{F,Any}, leafs, branches, ::Val{N}) where {F,N}
    quote
        Base.@ntuple $N i-> begin
            @inline
            iself_ref[]       += 1 
            children           = iparallel_childs .+ (i - 1)
            corrected_children = correct_children(fns_own_global[i], branches, children)
            CompositeEquation(iparent, corrected_children, iself_ref[], fns_own_global[i], leafs, Val(true))
        end
    end
end

@generated function add_global_equations(iparent, iparallel_childs, iself_ref, fns_own_global::F, leafs, branches, ::Val{N}) where {F,N}
    quote
        Base.@ntuple $N i-> begin
            @inline
            iself_ref[]       += 1
            children           = iparallel_childs .+ (i - 1)
            corrected_children = correct_children(fns_own_global[i], branches, children)
            CompositeEquation(iparent, corrected_children, iself_ref[], fns_own_global, leafs, Val(true))
        end
    end
end

function add_global_equations(iparent, iparallel_childs, iself_ref, fns_own_global::F, leafs, branches, ind_input, ::Val{B}, ::Val{1}) where {F,B}
    @inline
    iself_ref[]       += 1
    children           = iparallel_childs
    corrected_children = correct_children(fns_own_global, branches, children)
    CompositeEquation(iparent, corrected_children, iself_ref[], fns_own_global, leafs, ind_input, Val(B))
end

@generated function add_local_equations(iparent, ilocal_childs, iself_ref, fns_own_local, leafs, ::Val{N}) where {N}
    quote
        Base.@ntuple $N i-> begin
            @inline
            iself_ref[] += 1
            CompositeEquation(iparent, ilocal_childs[i], iself_ref[], fns_own_local[i], leafs, 0, Val(false))
        end
    end
end

@generated function add_parallel_equations(global_eqs::NTuple{N1,Any}, branches::NTuple{N2, AbstractCompositeModel}, iself_ref, fns_own_global::NTuple{N1,Any}) where {N1, N2}
    quote
        Base.@ntuple $N1 j -> begin
            @inline
            iparent_new = global_eqs[j].self
            fn          = counterpart(fns_own_global[j])
            Base.@ntuple $N2 i -> begin
                @inline
                generate_equations(branches[i], fn, isvolumetric(branches[i]); iparent = iparent_new, iself = iself_ref[])
            end
        end
    end
end

@generated function generate_args_template(eqs::NTuple{N, CompositeEquation}) where N 
    quote
        Base.@ntuple $N i -> differentiable_kwargs(eqs[i].fn) 
    end
end

@generated function generate_args_template(eqs::NTuple{N, Any}, x::SVector{N}, others::NamedTuple) where N
    quote
        args_template = generate_args_template(eqs)
        Base.@ntuple $N i -> begin
            @inline
            name =  keys(args_template[i])
            merge(NamedTuple{name}(x[i]), others)
        end
    end
end

@inline function evaluate_state_function(eq::CompositeEquation, args) 
    (; fn, rheology) = eq
    evaluate_state_function(fn, rheology, args)
end

@generated function evaluate_state_function(fn::F, rheology::NTuple{N, AbstractRheology}, args)  where {N,F}
    quote
        @inline
        vals = Base.@ntuple $N i -> fn(rheology[i], args)
        sum(vals)
    end
end

evaluate_state_function(fn::F, rheology::Tuple{}, args) where {F} = 0e0

# @inline evaluate_state_functions(eqs::NTuple{N, CompositeEquation}, args) where N = promote(ntuple(i -> evaluate_state_function(eqs[i], args[i]), Val(N))...)
@generated function evaluate_state_functions(eqs::NTuple{N, CompositeEquation}, args) where N 
    quote
        @inline
        Base.@ntuple $N i -> evaluate_state_function(eqs[i], args[i])
    end
end

add_child(::SVector, ::Tuple{}) = 0e0
# @generated function add_child(x::SVector{M}, child::NTuple{N}) where {M, N}
#     quote
#         @inline
#         v = Base.@ntuple $N i -> begin
#             ind = child[i]
#             M > ind ? x[ind] : 0e0
#         end
#         sum(v)
#     end
# end

@generated function add_child(x::SVector{M}, child::NTuple{N}) where {M, N}
    quote
        @inline
        v = Base.@ntuple $N i -> begin
            x[child[i]] 
        end
        sum(v)
    end
end

@generated function add_children(residual::NTuple{N, Any}, x::SVector{N}, eqs::NTuple{N, CompositeEquation}) where {N}
    quote
        @inline
        Base.@ntuple $N i -> residual[i] + add_child(x, eqs[i].child)
    end
end

function add_children(residual::Number, x::SVector, eq::CompositeEquation)
    residual + add_child(x, eq.child)
end

# if global, subtract the variables
@inline subtract_parent( ::SVector, eq::CompositeEquation{true} ,         vars) = vars[eq.ind_input]
@inline subtract_parent(x::SVector, eq::CompositeEquation{false}, ::NamedTuple) = x[eq.parent]
@generated function subtract_parent(residual::NTuple{N,Any}, x, eqs::NTuple{N, CompositeEquation}, vars) where {N} 
    quote
        @inline
        Base.@ntuple $N i -> residual[i] - subtract_parent(x, eqs[i], vars)
    end
end

function subtract_parent(residual::Number, x::SVector, eq::CompositeEquation, vars)
    residual - subtract_parent(x, eq, vars)
end
    
function compute_residual(c, x::SVector{N,T}, vars, others) where {N,T}
    
    eqs      = generate_equations(c)
    args_all = generate_args_template(eqs, x, others)

    # # evaluates the self-components of the residual
    residual = evaluate_state_functions(eqs, args_all)
    residual = add_children(residual, x, eqs)
    residual = subtract_parent(residual, x, eqs, vars)
    
    return SA[residual...]
end

function compute_residual(c, x::SVector{N,T}, vars, others, ind::Int64, ipartial::Int64) where {N,T}
    
    eqs      = generate_equations(c)
    args_all = generate_args_template(eqs, x, others)[1]

    # # evaluates the self-components of the residual
    eq       = eqs[1]
    residual = evaluate_state_function(eq, args_all)
    residual = add_children(residual, x, eq)
    residual = subtract_parent(residual, x, eq, vars)
    
    return residual
end
