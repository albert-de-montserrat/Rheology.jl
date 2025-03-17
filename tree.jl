struct Foo{T1, T2, F, R}
    parent::T1 # i-th element of x to be substracted
    child::T2  # i-th element of x to be added
    fn::F 
    rheology::R
end

branch_number(::Tuple{}, ::Int) = nothing
branch_number(::Any, i::Int) = i + 1

function foo(c::SeriesModel)
    (; branches) = c

    foo(branches[1], 1)
end

foo(::Tuple{}, ::Int) = ()

function foo(c::AbstractCompositeModel, iparent::Int)
    (; branches) = c
    a = Foo(i, branch_number(branches, iiparent))
    b = foo(branches, iparent + 1)
    a, b...
end

get_own_functions(c::NTuple{N, AbstractCompositeModel}) where N = ntuple(i -> get_own_functions(c[i]), Val(N))

function get_own_functions(c::SeriesModel)
    fns_own_all    = series_state_functions(c.leafs)
    fns_own_global = global_series_state_functions(fns_own_all)
    fns_own_local  = local_series_state_functions(fns_own_all)
    fns_own_global, fns_own_local
end

function get_own_functions(c::ParallelModel)
    fns_own_all    = parallel_state_functions(c.leafs)
    fns_own_global = global_parallel_state_functions(fns_own_all) |> flatten_repeated_functions |> superflatten
    fns_own_local  = local_parallel_state_functions(fns_own_all) # TODO
    fns_own_global, fns_own_local
end

add_child(::Tuple{}, ::Any) = nothing
add_child(::Any, i::Int) = i + 1
add_child(::Any, i::Base.RefValue{Int64}) = (i[] += 1; i)

foo(::Tuple{}; iparent = 0) = ()

function foo(c::AbstractCompositeModel; iparent = 1)
    (; branches, leafs) = c

    # iparent = 1
    
    fns_own_global, fns_own_local = get_own_functions(c)
    fns_branches_global,          = get_own_functions(branches)

    nown             = length(fns_own_global)
    nlocal           = length(fns_own_local)
    nbranches        = length(branches)

    iglobal          = ntuple(i -> iparent + i - 1, Val(nown))
    ilocal_childs    = ntuple(i -> iparent + nown - 1 + i, Val(nlocal))
    offsets_parallel = (0, length.(fns_branches_global)...)
    iparallel_childs = ntuple(i -> iparent + nlocal + offsets_parallel[i] + i, Val(nbranches))

    # add globals
    global_eqs = ntuple(Val(nown)) do i
        Foo(iparent, iglobal[i], fns_own_global[i], leafs)
    end

    # add locals
    local_eqs = ntuple(Val(nlocal)) do i
        Foo(iparent, ilocal_childs[i], fns_own_local[i], leafs)
    end

    # # add parallels
    # parallel_eqs = ntuple(Val(nbranches)) do i
    #     @inline
    #     branchᵢ = branches[i]
    #     fnᵢ     = fns_parallel_global[i]
    #     nfnᵢ    = length(fnᵢ)
    #     ntuple(Val(nfnᵢ)) do j
    #         @inline
    #         Foo(iparent, iparallel_childs[i], fnᵢ[j], branchᵢ.leafs)
    #     end
    # end

    nested_boys = foo(branches; iparent = iparent + nown + nlocal + 1)
    (global_eqs..., local_eqs..., nested_boys) |> superflatten
end

c = c0
# c = c.branches[1]
# nested_boys = foo(branches[1]; iparent = iparent + nown + nlocal + 1)

eqs = foo(c)
@code_warntype foo(c)
@b foo($c)
