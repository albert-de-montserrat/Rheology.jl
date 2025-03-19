include("numbering.jl")

struct CompositeEquation{IsGlobal, T1, T2, F, R}
    parent::T1  # i-th element of x to be substracted
    child::T2   # i-th element of x to be added
    self::Int64 # equation number
    fn::F       # state function
    rheology::R 

    function CompositeEquation(parent::T1, child::T2, self::Int64, fn::F, rheology::R, ::Val{B}) where {T1, T2, F, R, B}
        @assert B isa Bool
        new{B, T1, T2, F, R}(parent, child, self, fn, rheology)
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
get_own_functions(c::SeriesModel)   = get_own_functions(c, series_state_functions, global_series_state_functions, local_series_state_functions)
get_own_functions(c::ParallelModel) = get_own_functions(c, parallel_state_functions, global_parallel_state_functions, local_parallel_state_functions)

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

generate_equations(::Tuple{}; iparent = 0) = ()

function generate_equations(c::AbstractCompositeModel; iparent = 0, iself = 0)
    iself_ref = Ref{Int64}(iself)

    (; branches, leafs) = c
    
    fns_own_global, fns_own_local = get_own_functions(c)
    fns_branches_global,          = get_own_functions(branches)

    nown             = length(fns_own_global)
    nlocal           = length(fns_own_local)
    nbranches        = length(branches)

    iglobal          = ntuple(i -> iparent + i - 1, Val(nown))
    ilocal_childs    = ntuple(i -> iparent + nown - 1 + i, Val(nlocal))
    offsets_parallel = (0, length.(fns_branches_global)...)
    iparallel_childs = ntuple(i -> iparent + nlocal + offsets_parallel[i] + i + nown, Val(nbranches))

    # add global equations
    global_eqs = ntuple(Val(nown)) do i
        iself_ref[] += 1
        get_local_functions(branches)
        CompositeEquation(iparent, iparallel_childs .+ (i - 1), iself_ref[], fns_own_global[i], leafs, Val(true))
    end 
    
    # add local equations
    local_eqs = ntuple(Val(nlocal)) do i
        iself_ref[] += 1
        CompositeEquation(iparent, ilocal_childs[i], iself_ref[], fns_own_local[i], leafs, Val(false))
    end

    parallel_eqs = ntuple(Val(nown)) do j
        iparent_new = global_eqs[j].self
        fn          = counterpart(fns_own_global[j])
        ntuple(Val(nbranches)) do i
            generate_equations(branches[i], fn; iparent = iparent_new, iself = iself_ref[])
        end
    end
    
    (global_eqs..., local_eqs..., parallel_eqs...) |> superflatten
end

function generate_equations(c::AbstractCompositeModel, fns_own_global::F; iparent = 0, iself = 0) where F
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
    global_eqs = ntuple(Val(nown)) do i
        iself_ref[] += 1
        get_local_functions(branches)
        CompositeEquation(iparent, iparallel_childs .+ (i - 1), iself_ref[], fns_own_global, leafs, Val(false))
    end 

    # add locals
    local_eqs = ntuple(Val(nlocal)) do i
        iself_ref[] += 1
        CompositeEquation(iparent, ilocal_childs[i], iself_ref[], fns_own_local[i], leafs, Val(false))
    end

    parallel_eqs = ntuple(Val(nown)) do j
        iparent_new = global_eqs[j].self
        fn          = counterpart(fns_own_global)
        ntuple(Val(nbranches)) do i
            generate_equations(branches[i], fn; iparent = iparent_new, iself = iself_ref[])
        end
    end
    
    (global_eqs..., local_eqs..., parallel_eqs...) |> superflatten
end

@inline generate_args_template(eqs::NTuple{N, CompositeEquation}) where N = ntuple(i -> differentiable_kwargs(eqs[i].fn), Val(N))

function generate_args_template(eqs::NTuple{N, Any}, x::SVector{N}, others::NamedTuple) where N
    args_template = generate_args_template(eqs)
    ntuple(Val(N)) do i
        @inline
        name =  keys(args_template[i])
        merge(NamedTuple{name}(x[i]), others)
    end
end

@inline function evaluate_state_function(eq::CompositeEquation, args) 
    (; fn, rheology) = eq      
    N = length(rheology)
    vals = ntuple(Val(N)) do i
        fn(rheology[i], args)
    end
    sum(vals)
end

@inline evaluate_state_functions(eqs::NTuple{N, CompositeEquation}, args) where N = promote(ntuple(i -> evaluate_state_function(eqs[i], args[i]), Val(N))...)

@inline add_child( ::SVector{N,T}, ::Tuple{}) where {N,T} = zero(T)
@inline add_child(x::SVector, child::NTuple)              = sum(@inbounds x[i] for i in child)

@inline function add_children(residual::NTuple{N,T}, x::SVector{N,T}, eqs::NTuple{N, CompositeEquation}) where {N,T}
    ntuple(Val(N)) do i
        residual[i] + add_child(x, eqs[i].child)
    end
end
 
# if global, subtract the variables
@inline subtract_parent( ::SVector, eq::CompositeEquation{true} ,         vars) = vars[eq.self]
@inline subtract_parent(x::SVector, eq::CompositeEquation{false}, ::NamedTuple) = x[eq.parent]
@inline subtract_parent(residual::NTuple{N,T}, x, eqs::NTuple{N, CompositeEquation}, vars) where {N,T} = ntuple(i -> residual[i] - subtract_parent(x, eqs[i], vars), Val(N))

function compute_residual(c, x, vars, others)
    eqs      = generate_equations(c)
    args_all = generate_args_template(eqs, x, others)

    # evaluates the self-components of the residual
    residual = evaluate_state_functions(eqs, args_all)
    residual = add_children(residual, x, eqs)
    residual = subtract_parent(residual, x, eqs, vars)
    SA[residual...]
end
@code_warntype compute_residual(c, x, vars, others)

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)

composite  = viscous1, powerlaw

c, x, vars, args, others = let
    # elastic - viscous -- parallel
    #                         |  
    #                viscous --- viscous  
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

@code_warntype compute_residual(c, x, vars, others)

compute_residual(c, x, vars, others)
ForwardDiff.jacobian(x -> compute_residual(c, x, vars, others), x)

@b compute_residual($(c, x, vars, others)...)
ForwardDiff.jacobian(x -> compute_residual(c, x, vars, others), x)