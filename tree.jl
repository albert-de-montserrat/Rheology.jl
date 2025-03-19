include("numbering.jl")

# branch_number(::Tuple{}, ::Int) = nothing
# branch_number(::Any, i::Int) = i + 1

# function foo(c::SeriesModel)
#     (; branches) = c

#     foo(branches[1], 1)
# end

# foo(::Tuple{}, ::Int) = ()

# function foo(c::AbstractCompositeModel, iparent::Int)
#     (; branches) = c
#     a = CompositeEquation(i, branch_number(branches, iiparent))
#     b = foo(branches, iparent + 1)
#     a, b...
# end

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

struct CompositeEquation{T1, T2, F, R}
    parent::T1  # i-th element of x to be substracted
    child::T2   # i-th element of x to be added
    self::Int64 # equation number
    fn::F       # state function
    rheology::R
end

get_own_functions(c::NTuple{N, AbstractCompositeModel}) where N = ntuple(i -> get_own_functions(c[i]), Val(N))

function get_own_functions(c::SeriesModel)
    fns_own_all    = series_state_functions(c.leafs)
    fns_own_global = global_series_state_functions(fns_own_all) |> superflatten |> flatten_repeated_functions
    fns_own_local  = local_series_state_functions(fns_own_all)
    fns_own_global, fns_own_local
end

function get_own_functions(c::ParallelModel)
    fns_own_all    = parallel_state_functions(c.leafs)
    fns_own_global = global_parallel_state_functions(fns_own_all) |> superflatten |> flatten_repeated_functions
    fns_own_local  = local_parallel_state_functions(fns_own_all)
    fns_own_global, fns_own_local
end

get_local_functions(c::NTuple{N, AbstractCompositeModel}) where N = ntuple(i -> get_own_functions(c[i]), Val(N))

function get_local_functions(c::SeriesModel)
    fns_own_all    = series_state_functions(c.leafs)
    local_series_state_functions(fns_own_all)
end

function get_local_functions(c::ParallelModel)
    fns_own_all    = parallel_state_functions(c.leafs)
    local_parallel_state_functions(fns_own_all)
end

get_own_functions(::Tuple{}) = (), ()

# x,y=get_own_functions(b)
# b = c.branches[1].branches
# add_child(::Tuple{}, ::Any) = nothing
# add_child(::Any, i::Int) = i + 1
# add_child(::Any, i::Base.RefValue{Int64}) = (i[] += 1; i)

foo(::Tuple{}; iparent = 0) = ()

iparent = 0; ichild = 0; iself = 0
function foo(c::AbstractCompositeModel; iparent = 0, iself = 0)
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
        CompositeEquation(iparent, iparallel_childs .+ (i - 1), iself_ref[], fns_own_global[i], leafs)
    end 

    # add local equations
    local_eqs = ntuple(Val(nlocal)) do i
        iself_ref[] += 1
        CompositeEquation(iparent, ilocal_childs[i], iself_ref[], fns_own_local[i], leafs)
    end

    parallel_eqs = ntuple(Val(nown)) do j
        iparent = global_eqs[j].self
        fn      = counterpart(fns_own_global[j])
        ntuple(Val(nbranches)) do i
            # foo(branches[i]; iparent = iparent + nown + nlocal, iself = iself_ref[])
            foo(branches[i], fn; iparent = iparent, iself = iself_ref[])
        end
    end
    
    (global_eqs..., local_eqs..., parallel_eqs...) |> superflatten
end

function foo(c::AbstractCompositeModel, fns_own_global::F; iparent = 0, iself = 0) where F
    iself_ref = Ref{Int64}(iself)

    (; branches, leafs) = c
    
    _, fns_own_local      = get_own_functions(c)
    fns_branches_global,_ = get_own_functions(branches)

    nown             = 1 #length(fns_own_global)
    nlocal           = length(fns_own_local)
    nbranches        = length(branches)

    iglobal          = ntuple(i -> iparent + i - 1, Val(nown))
    ilocal_childs    = ntuple(i -> iparent + nown - 1 + i, Val(nlocal))
    offsets_parallel = (0, length.(fns_branches_global)...)
    iparallel_childs = ntuple(i -> iparent + nlocal + offsets_parallel[i] + i + nown, Val(nbranches))

    # add globals
    global_eqs = ntuple(Val(nown)) do i
        iself_ref[] += 1
        get_local_functions(branches)
        CompositeEquation(iparent, iparallel_childs .+ (i - 1), iself_ref[], fns_own_global, leafs)
    end 

    # add locals
    local_eqs = ntuple(Val(nlocal)) do i
        iself_ref[] += 1
        CompositeEquation(iparent, ilocal_childs[i], iself_ref[], fns_own_local[i], leafs)
    end

    @show global_eqs
    parallel_eqs = ntuple(Val(nown)) do j
        iparent = global_eqs[j].self
        fn      = counterpart(fns_own_global)
        ntuple(Val(nbranches)) do i
            # foo(branches[i]; iparent = iparent + nown + nlocal, iself = iself_ref[])
            foo(branches[i], fn; iparent = iparent, iself = iself_ref[])
        end
    end
    
    (global_eqs..., local_eqs..., parallel_eqs...) |> superflatten
end

# @code_warntype foo(c)
eqs = foo(c)
@show length(eqs)

i = 4; println(
"
    self     => $(eqs[i].self)
    parent   => $(eqs[i].parent)
    child    => $(eqs[i].child)
    fn       => $(eqs[i].fn)
    rheology => $(eqs[i].rheology)
"
)

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)

composite  = viscous1, powerlaw

c, x, vars, args, others = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    s1     = SeriesModel(viscous1, viscous2)
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

c, x, vars, args, others = let
    # elastic - viscous -- parallel
    #                         |  
    #                viscous --- viscous  
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(viscous1, viscous2)
    c      = SeriesModel(elastic, viscous1, p)
    vars   = (; ε = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

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

eqs = foo(c)

for eq in eqs
println(
"
    parent   => $(eq.parent)
    child    => $(eq.child)
    fn       => $(eq.fn)
    rheology => $(eq.rheology)
"
)
end