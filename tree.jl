include("numbering.jl")

struct CompositeEquation{IsGlobal, T, F, R}
    parent::Int64  # i-th element of x to be substracted
    child::T       # i-th element of x to be added
    self::Int64    # equation number
    fn::F          # state function
    rheology::R 

    function CompositeEquation(parent::Int64, child::T, self::Int64, fn::F, rheology::R, ::Val{B}) where {T, F, R, B}
        @assert B isa Bool
        new{B, T, F, R}(parent, child, self, fn, rheology)
    end
end

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

get_local_functions(c::NTuple{N, AbstractCompositeModel}) where N = ntuple(i -> get_own_functions(c[i]), Val(N))

function get_local_functions(c::SeriesModel)
    fns_own_all    = series_state_functions(c.leafs)
    local_series_state_functions(fns_own_all)
end

function get_local_functions(c::ParallelModel)
    fns_own_all    = parallel_state_functions(c.leafs)
    local_parallel_state_functions(fns_own_all)
end

@inline generate_equations(::Tuple{}; iparent = 0) = ()

# function generate_equations(c::AbstractCompositeModel; iparent::Int64 = 0, iself::Int64 = 0)
#     iself_ref = Ref{Int64}(iself)

#     (; branches, leafs) = c
    
#     fns_own_global, fns_own_local = get_own_functions(c)
#     fns_branches_global,          = get_own_functions(branches)

#     nown             = length(fns_own_global)
#     nlocal           = length(fns_own_local)
#     nbranches        = length(branches)

#     # iglobal          = ntuple(i -> iparent + i - 1, Val(nown))
#     ilocal_childs    = ntuple(i -> iparent + nown - 1 + i, Val(nlocal))
#     offsets_parallel = (0, length.(fns_branches_global)...)
#     iparallel_childs = ntuple(i -> iparent + nlocal + offsets_parallel[i] + i + nown, Val(nbranches))

#     # add global equations
#     global_eqs = ntuple(Val(nown)) do i
#         @inline
#         iself_ref[] += 1
#         CompositeEquation(iparent, iparallel_childs .+ (i - 1), iself_ref[], fns_own_global[i], leafs, Val(true))
#     end 

    
#     # add local equations
#     local_eqs = ntuple(Val(nlocal)) do i
#         @inline
#         iself_ref[] += 1
#         CompositeEquation(iparent, ilocal_childs[i], iself_ref[], fns_own_local[i], leafs, Val(false))
#     end

#     parallel_eqs = ntuple(Val(nown)) do j
#         @inline
#         iparent_new = global_eqs[j].self
#         fn          = counterpart(fns_own_global[j])
#         ntuple(Val(nbranches)) do i
#             @inline
#             generate_equations(branches[i], fn; iparent = iparent_new, iself = iself_ref[])
#         end
#     end
    
#     (global_eqs..., local_eqs..., parallel_eqs...) |> superflatten
# end

@generated function add_global_equations(iparent, iparallel_childs, iself_ref, fns_own_global::NTuple{F,Any}, leafs, ::Val{N}) where {F,N}
    quote
        Base.@ntuple $N i-> begin
            @inline
            iself_ref[] += 1
            CompositeEquation(iparent, iparallel_childs .+ (i - 1), iself_ref[], fns_own_global[i], leafs, Val(true))
        end
    end
end

@generated function add_global_equations(iparent, iparallel_childs, iself_ref, fns_own_global::F, leafs, ::Val{N}) where {F,N}
    quote
        Base.@ntuple $N i-> begin
            @inline
            iself_ref[] += 1
            CompositeEquation(iparent, iparallel_childs .+ (i - 1), iself_ref[], fns_own_global, leafs, Val(true))
        end
    end
end


@generated function add_local_equations(iparent, ilocal_childs, iself_ref, fns_own_local, leafs, ::Val{N}) where {N}
    quote
        Base.@ntuple $N i-> begin
            @inline
            iself_ref[] += 1
            CompositeEquation(iparent, ilocal_childs[i], iself_ref[], fns_own_local[i], leafs, Val(false))
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
                generate_equations(branches[i], fn; iparent = iparent_new, iself = iself_ref[])
            end
        end
    end
end

function generate_equations(c::AbstractCompositeModel; iparent::Int64 = 0, iself::Int64 = 0)
    iself_ref = Ref{Int64}(iself)

    (; branches, leafs) = c
    
    fns_own_global, fns_own_local = get_own_functions(c)
    fns_branches_global,          = get_own_functions(branches)

    nown             = length(fns_own_global)
    nlocal           = length(fns_own_local)
    nbranches        = length(branches)

    ilocal_childs    = ntuple(i -> iparent + nown - 1 + i, Val(nlocal))
    offsets_parallel = (0, length.(fns_branches_global)...)
    iparallel_childs = ntuple(i -> iparent + nlocal + offsets_parallel[i] + i + nown, Val(nbranches))

    # add global equations
    global_eqs = add_global_equations(iparent, iparallel_childs, iself_ref, fns_own_global, leafs, Val(nown))
    # add local equations
    local_eqs =  add_local_equations(iparent, ilocal_childs, iself_ref, fns_own_local, leafs, Val(nlocal))
    # add parallel equations
    parallel_eqs = add_parallel_equations(global_eqs, branches, iself_ref, fns_own_global)
    
    return (global_eqs..., local_eqs..., parallel_eqs...) |> superflatten 
end

function generate_equations(c::AbstractCompositeModel, fns_own_global::F; iparent::Int64 = 0, iself::Int64 = 0) where F
    iself_ref = Ref{Int64}(iself)

    (; branches, leafs) = c
    
    _, fns_own_local      = get_own_functions(c)
    # fns_branches_global,_ = get_own_functions(branches)

    nown             = 1 #length(fns_own_global)
    nlocal           = length(fns_own_local)
    nbranches        = length(branches)

    # iglobal          = ntuple(i -> iparent + i - 1, Val(nown))
    ilocal_childs    = ntuple(i -> iparent + nown - 1 + i, Val(nlocal))
    offsets_parallel = (0, ntuple(i -> i, Val(nbranches))...)
    # offsets_parallel = (0, length.(fns_branches_global)...)
    iparallel_childs = ntuple(i -> iparent + nlocal + offsets_parallel[i] + i + nown, Val(nbranches))

    # add globals
    iself_ref[] += 1
    global_eqs   = CompositeEquation(iparent, iparallel_childs, iself_ref[], fns_own_global, leafs, Val(false))
    
    # global_eqs = ntuple(Val(nown)) do i
        # @inline
        # iself_ref[] += 1
        # CompositeEquation(iparent, iparallel_childs .+ (i - 1), iself_ref[], fns_own_global, leafs, Val(false))
    # end 

    local_eqs =  add_local_equations(iparent, ilocal_childs, iself_ref, fns_own_local, leafs, Val(nlocal))

    # parallel_eqs = ntuple(Val(nown)) do j
    #     @inline
    #     iparent_new = global_eqs[j].self
    #     fn          = counterpart(fns_own_global[j])
    #     ntuple(Val(nbranches)) do i
    #         @inline
    #         generate_equations(branches[i], fn; iparent = iparent_new, iself = iself_ref[])
    #     end
    # end
    
    iparent_new  = global_eqs.self
    fn           = counterpart(fns_own_global)
    parallel_eqs = ntuple(Val(nbranches)) do i
        @inline
        generate_equations(branches[i], fn; iparent = iparent_new, iself = iself_ref[])
    end

    return (global_eqs, local_eqs..., parallel_eqs...) |> superflatten
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

@generated function evaluate_state_function(fn::F,rheology::NTuple{N, AbstractRheology}, args)  where {N,F}
    quote
        @inline
        vals = Base.@ntuple $N i -> fn(rheology[i], args)
        sum(vals)
    end
end

# @inline evaluate_state_functions(eqs::NTuple{N, CompositeEquation}, args) where N = promote(ntuple(i -> evaluate_state_function(eqs[i], args[i]), Val(N))...)
@generated function evaluate_state_functions(eqs::NTuple{N, CompositeEquation}, args) where N 
    quote
        @inline
        Base.@nexprs $N i -> v_i = evaluate_state_function(eqs[i], args[i])
        Base.@ncall $N tuple v
    end
end

add_child( ::SVector, ::Tuple{}) = 0e0
@generated function add_child(x::SVector, child::NTuple{N}) where {N}
    quote
        @inline
        v = Base.@ntuple $N i -> x[child[i]]
        sum(v)
    end
end

@generated function add_children(residual::NTuple{N, Any}, x::SVector{N}, eqs::NTuple{N, CompositeEquation}) where {N}
    quote
        @inline
        Base.@ntuple $N i -> residual[i] + add_child(x, eqs[i].child)
    end
end
 
# if global, subtract the variables
@inline subtract_parent( ::SVector, eq::CompositeEquation{true} ,         vars) = vars[eq.self]
@inline subtract_parent(x::SVector, eq::CompositeEquation{false}, ::NamedTuple) = x[eq.parent]
@generated function subtract_parent(residual::NTuple{N,Any}, x, eqs::NTuple{N, CompositeEquation}, vars) where {N} 
    quote
        @inline
        Base.@ntuple $N i -> residual[i] - subtract_parent(x, eqs[i], vars)
    end
end

function compute_residual(c, x::SVector{N,T}, vars, others) where {N,T}
    
    eqs      = generate_equations(c)
    args_all = generate_args_template(eqs, x, others)

    # # evaluates the self-components of the residual
    residual =  evaluate_state_functions(eqs, args_all)
    residual =  add_children(residual, x, eqs)
    residual =  subtract_parent(residual, x, eqs, vars)
    
    return SA[residual...]
end

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)

c, x, vars, args, others = let
    # elastic - viscous -- parallel
    #                         |  
    #                viscous --- viscous  |
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(viscous1, viscous2)
    c      = SeriesModel(elastic, viscous1, p)
    vars   = (; ε = 1e-15, θ = 1e-20) # input variables (constant)
    args   = (; τ = 1e2, P = 1e6)     # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)            # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c, x, vars, args, others = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, p)
    vars   = (; ε = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(args)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c, x, vars, args, others = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    p      = ParallelModel(viscous1, viscous2)
    c      = SeriesModel(viscous1, p)
    vars   = (; ε = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

compute_residual(c, x, vars, others)
ForwardDiff.jacobian(x -> compute_residual(c, x, vars, others), x)

# @b compute_residual($(c, x, vars, others)...)
# @b ForwardDiff.jacobian(y -> compute_residual($c, y, $vars, $others), $x)