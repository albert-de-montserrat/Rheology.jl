# Test file to evaluate some concepts that B has in mind

using LinearAlgebra
using StaticArrays
using ForwardDiff
using Test

include("composite.jl")
include("rheology_types.jl")
include("state_functions.jl")
include("matrices.jl")
#include("others.jl")

include("kwargs.jl")




# The strainrate is required for this element


args = (; τ = 1e2, P = 1e6, dt = 1e10) # we solve for this, initial guess
vars = (; ε = 1e-15) # input variables


# example 1, paragraph 6
elastic_i = IncompressibleElasticity(1e10);
viscous   = LinearViscosity(1e18);
powerlaw  = PowerLawViscosity(5e19, 3);
drucker   = DruckerPrager(1e6, 30, 0)
p         = ParallelModel(viscous, powerlaw);
s         = SeriesModel(elastic_i, p);
c         = CompositeModel(s);
diff_args, res_args  = get_all_kwargs(c.components) 

#=
# example 2, paragraph 6
s31       = SeriesModel(viscous, elastic_i);     
p3        = ParallelModel(s31, viscous);
s         = SeriesModel(viscous, elastic_i, p3);
c         = CompositeModel(s);
diff_args, res_args  = get_all_kwargs(c.components)
=#


# define a serial component consisting of rheologies that only have a compute_stress function defined:
viscous_s =  LinearViscosityStress(1e19);
viscous   =  LinearViscosity(1e19);
s         = SeriesModel(viscous_s, viscous_s, viscous_s);
s1        = SeriesModel(viscous, viscous, viscous);
s2        = SeriesModel(viscous, viscous_s, viscous);

#diff_args, res_args  = get_all_kwargs(s1)

#flatten_repeated_functions(statefuns)


# returns a vector with kwargs for a serial element along with their number
function get_kwargs_vec(s::SeriesModel{N}) where N

    # This allocates -> the sizes of the vectors are known when the element is defined, though,
    #  so we may be able to preallocate stuff 
    kwargs_vec   = Vector{NamedTuple}(undef,N+1)

    if isvolumetric(s)
        model_args = (; τ = 0.0, P = 0.0)
    else
        model_args = (; τ = 0.0,)
    end
    model_keys    = keys(model_args)
    kwargs_vec[1] = model_args

    for i = 1:N
        funs  = series_state_functions(s[i])
        kwargs = differentiable_kwargs(funs)    # contains all the variables for this element
        kwargs_vec[i+1]     = kwargs
    end

    
    # At this stage, kwargs_vec contains all keywords for all elements.
    # Some, such as :τ and :P are repeated and should be removed 
    all_keys = model_keys
    all_elements = (fill(s.n[1],length(model_keys))...,)
    all_model    = (fill(s.parent[1],length(model_keys))...,)
    for i=2:N+1
        keys_el = keys(kwargs_vec[i])
        for (_,key) in enumerate(keys_el)
            if key != :τ && key != :P
                all_keys = (all_keys..., key)
                all_elements = (all_elements..., s.num[i-1])
                all_model = (all_model..., s.parent[1])
            end
        end
    end
  
    return kwargs_vec, all_keys, all_elements, all_model
end


kwargs_vec, all_keys, all_elements, all_model = get_kwargs_vec(s1)


vars       = (; ε = 1e-15)  # input variables
args_solve = (; τ = 1e2)    # we solve for this, initial guess
args_other = (; dt = 1e10)  # other args that may be needed, non differentiable
funs_local = series_state_functions(s.children)
args_local = all_differentiable_kwargs(funs_local)



#R = compute_residual(c.components.children, statefuns, vars, args)
#J = compute_jacobian(x, composite, statefuns, args_diff, args_nondiff)




# ==
# Albert's suggestion:
#=
viscous    = LinearViscosity(5e19)
powerlaw   = PowerLawViscosity(5e19, 3)
elastic    = Elasticity(1e10, 1e12) # im making up numbers
dt         = 1e10
vars       = (; ε = 1e-15, θ = 1e-20) # input variables
args_solve = (; τ = 1e2, P = 1e6) # we solve for this, initial guess

args_other = (; dt = 1e10) # other args that may be needed, non differentiable
#composite = viscous, powerlaw, elastic
#composite = viscous_s, viscous_s, viscous_s
composite = viscous, viscous, viscous

#funs_local     = parallel_state_functions(composite)
funs_local     = series_state_functions(composite)

args_local     = all_differentiable_kwargs(funs_local)

# we can modfy this if needed to have different initial conditions for different elements
args_local_aug = ntuple(Val(length(args_local))) do i 
    merge(args_local[i], args_other)
end

unique_funs_local = flatten_repeated_functions(funs_local)
N_reductions      = length(unique_funs_local)
state_reductions  = ntuple(i -> state_var_reduction, Val(N_reductions))

state_funs        = merge_funs(state_reductions, funs_local)
reduction_ind     = reduction_funs_args_indices(funs_local, unique_funs_local)

# # split args into differentiable and not differentiable
# args_diff, args_nondiff = split_args(args, statefuns)
# # rhs of the system of eqs, initial guess
# x = SA[values(args_diff)...]

local_x = SA[Base.IteratorsMD.flatten(values.(args_local))...]
state_x = SA[values(args_solve)...]
x       = SA[state_x..., local_x...]

args_state = ntuple(Val(N_reductions)) do i
    ntuple(Val(length(reduction_ind[i]))) do j
        ind = reduction_ind[i][j]
        local_x[ind]
    end
end
=#


#=
# Albert, v2

viscous    = LinearViscosity(5e19)
powerlaw   = PowerLawViscosity(5e19, 3)
elastic    = Elasticity(1e10, 1e12) # im making up numbers
composite = viscous, powerlaw, elastic

dt         = 1e10
vars       = (; ε = 1e-15, θ = 1e-20) # input variables
args_solve = (; τ = 1e2, P = 1e6) # we solve for this, initial guess
args_other = (; dt = 1e10) # other args that may be needed, non differentiable


funs_local     = parallel_state_functions(composite)
args_local     = all_differentiable_kwargs(funs_local)
args_local_aug = ntuple(Val(length(args_local))) do i 
    merge(args_local0[i], args_other)
end

unique_funs_local = flatten_repeated_functions(funs_local)
N_reductions      = length(unique_funs_local)
state_reductions  = ntuple(i -> state_var_reduction, Val(N_reductions))
args_reduction    = ntuple(_ -> (), Val(N_reductions))
state_funs        = merge_funs(state_reductions, funs_local)
reduction_ind     = reduction_funs_args_indices(funs_local, unique_funs_local)

inds_args_to_x    = tuple(
    tuple([ind .+ N_reductions for ind in reduction_ind]...)...,
    ntuple(x -> (x + N_reductions,), Val(length(funs_local)))...,
)

# # split args into differentiable and not differentiable
# args_diff, args_nondiff = split_args(args, statefuns)
# # rhs of the system of eqs, initial guess
# x = SA[values(args_diff)...]

local_x = SA[Base.IteratorsMD.flatten(values.(args_local))...]
state_x = SA[values(args_solve)...]
x       = SA[state_x..., local_x...]

args_state = ntuple(Val(N_reductions)) do i
    ntuple(Val(length(reduction_ind[i]))) do j
        ind = reduction_ind[i][j]
        local_x[ind]
    end
end


args_all      = tuple(args_state..., args_local_aug...)
args_template = tuple(args_reduction..., args_local...)


# jjacobian 

i = 1
compositeᵢ = composite[i]
f = x -> begin
    args_tmp = generate_args_from_x(x, inds_args_to_x, args_template, args_all)
    eval_state_functions(state_funs, compositeᵢ, args_tmp)
end
ForwardDiff.jacobian(x -> f(x), x)

=#


# ==