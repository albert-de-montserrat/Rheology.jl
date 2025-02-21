using LinearAlgebra
using StaticArrays
using ForwardDiff
using DifferentiationInterface

include("src/composite.jl") # not functional yet
include("src/rheology_types.jl")
include("src/state_functions.jl")
include("src/kwargs.jl")
include("src/matrices.jl")

function bt_line_search(Δx, J, R, statefuns, composite::NTuple{N, Any}, args, vars; α=1.0, ρ=0.5, c=1e-4, α_min=1e-8) where N
    perturbed_args = augment_args(args, α * Δx)
    perturbed_R    = compute_residual(composite, statefuns, vars, perturbed_args)
      
    while sqrt(sum(perturbed_R.^2)) > sqrt(sum((R + (c * α * (J * Δx))).^2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_args = augment_args(args, α * Δx)
        perturbed_R = compute_residual(composite, statefuns, vars, perturbed_args) 
    end
    return α
end

# compute the residual vector
function eval_residual(x, composite::NTuple{N,Any}, args_elements::NTuple{N,Any}, funs_elements::NTuple{N,Any}, entry_residual, vars, vars_ε, vars_θ, var_names, var_elems) where N
    res     = MArray(zero(x))
    res[1]  = -vars.ε
    if any(vars_θ) 
        #if iscompressible(composite)   # this line sometimes allocates
        res[2] = -vars.θ
    end

    # deal with strainrate and volumetric strainrate component
    for i=1:length(x)
        if vars_ε[i]
            res[1]  += x[i]
            res[i] = -x[1]
        end
        if vars_θ[i]
            res[2]  += x[i]
            res[i] = -x[i]
        end
    end

    # Evaluate the residual for each of the rheological elements and sum the result
    res += local_residual_series(funs_elements, composite, args_elements, entry_residual, x, var_elems, var_names, vars_ε, vars_θ)

    return SVector(res)
end

# evaluates the residual function and returns a SVector with the same size as x in an allocation-free manner
@generated function local_residual(statefuns::Function, composite_val::AbstractRheology, args_local::NamedTuple, iel::Int64, x::SVector{N,T}) where {N,T}
    quote
        f = Base.@ntuple $N i -> begin
           if i == iel 
                eval_state_function(statefuns, composite_val, args_local)
           else
                zero(T)
           end
        end
        SVector{$N, $T}(f)
    end
end

# Same but with a Tuple
@generated function local_residual(statefuns::NTuple{N1,Function}, composite_val::AbstractRheology, args_local::NamedTuple, iel::NTuple{N1,Int64}, x::SVector{N,T}) where {N1,N,T}
    quote
        v = SVector{N, T}(zero(x))
        Base.@nexprs $N1 k -> begin
            v += local_residual(statefuns[k], composite_val, args_local, iel[k], x) 
        end
        v
    end
end

# Same but with Tuple of Tuples
@generated function local_residual_series(  funs_elements::NTuple{N,Any}, 
                                            composite::NTuple{N,Any}, 
                                            args_elements::NTuple{N,Any}, 
                                            entry_residual::NTuple{N,Any}, 
                                            x::SVector{N1,T}, 
                                            var_elems::NTuple{N1,Int}, 
                                            var_names::NTuple{N1,Symbol}, 
                                            vars_ε::NTuple{N1,Bool}, 
                                            vars_θ::NTuple{N1,Bool}) where {N,N1,T}
    quote
        res = SVector{N1, T}(zero(x))
        Base.@nexprs $N iel -> begin
            args_local = args_elements[iel]

            # WE NEED TO AUGMENT THIS! 
            args_local = augment_args_local(args_local, var_elems, var_names, vars_ε, vars_θ, x, iel)


            res += local_residual(funs_elements[iel], composite[iel], args_local, entry_residual[iel], x) 
        end
        SVector(res)
    end
end


function augment_args_local(args_local::NamedTuple, var_elems::NTuple{N,Int}, var_names::NTuple{N,Symbol}, vars_ε::NTuple{N, Bool}, vars_θ::NTuple{N, Bool}, x::SVector{N,T}, el::Int)  where {N,T}
    # This changes the value within the args_local to the values of x
    # It does assume that the value of x is part of args_local
    
    k = keys(args_local)
    vals = MVector{length(args_local), eltype(x)}(values(args_local))
    for j = 1:N
        if (var_elems[j] == 0 || var_elems[j] == el) & !vars_ε[j] & !vars_θ[j] 
            id = findfirst(k .== var_names[j])
            if !isnothing(id)
                vals[id] = x[j] 
            end
        end
    end

    return NamedTuple{k}(vals)
end


function statefuns_to_residual_series(funs_elements::NTuple{N, Any}, n=1) where N
    # this allocates - to be fixed!
    el_local = ()
    for statefuns in funs_elements
        num = ()
        for i=1:length(statefuns)
            if statefuns[i] === compute_strain_rate
                num = (num..., 1)
            elseif statefuns[i] === compute_volumetric_strain_rate
                num = (num..., 2)    
            else
                num = (num..., n)
                n += 1
            end
        end
        el_local = (el_local..., num)
    end

    return el_local
end


function main(composite, vars, args_solve0, args_other)

    # differentiable arguments for each of the rheological elements within the composite
    funs_elements = series_state_functions.(composite)                      # state functions for each rheological element
    funs_unique   = get_unique_state_functions(composite, :series)          # unique state functions for the full serial element
    args_elements0= differentiable_kwargs.(funs_elements)                   # differentiable args for each element
    args_solve    = merge(differentiable_kwargs(funs_unique), args_solve0)  # differentiable args for the full serial element

    # add solution arguments
    args_elements    = ntuple(Val(length(args_elements0))) do i 
        merge(args_elements0[i], args_solve0)
    end

    # augment 
    args_elements_aug    = ntuple(Val(length(args_elements0))) do i 
        merge(args_elements[i], args_other)
    end

    # determine if we iterate for τ, for P or for both. 
    iteration_vars = (;)
    if isshear(composite)
        if haskey(args_solve,:τ)
            iteration_vars = merge(iteration_vars,(τ=args_solve.τ,))
        else
            iteration_vars = merge(iteration_vars,(τ=0,))
        end
    end
    if iscompressible(composite)
        if haskey(args_solve,:P)
            iteration_vars = merge(iteration_vars,(P=args_solve.P,))
        else
            iteration_vars = merge(iteration_vars,(P=0,))
        end
    end

    # Get the entry in the residual vector for every state function
    entry_residual = statefuns_to_residual_series(funs_elements, length(iteration_vars)+1)  # allocates - to be fixed

    # now remove the "iteration_vars" variables from args_elements, as we always have to solve for this in a serial element
    # this implies we are left with the "additional" variables we iterate for, for each rheological element (if any)
    args_local        = ntuple(Val(length(args_elements))) do i 
        Base.structdiff(args_elements[i], iteration_vars)
    end

    # additional variables for each rheological element (that are not τ or P)
    el_local = ntuple(Val(length(args_local))) do i 
        ntuple(j -> i, Val(length(args_local[i]))) 
    end

    # lets number each of these additional variables so we know 
    Nsol = length(iteration_vars)
    #num_vars = ntuple(Val(length(args_local))) do i 
    #    ntuple(j -> n+j, k=j, Val(length(args_local[i]))) 
    #   #Nsol += length(args_local[i])
    #end

    # flatten the additional variables (besides τ and P) for each rheological element
    # the trick here is that we may have several variables with the same name, so we cannot have a single NamedTuple for this
    # we also don't need that as we simply need to know which variable belongs to which rheological element
    N_vars      = sum(length.(args_local)) + Nsol;
    vars_all    = keys(iteration_vars)

    # Create vector with initial variable names. Note that we can have the same name multiple times
    var_names   = (keys(iteration_vars)..., Base.IteratorsMD.flatten(keys.(args_local))...)

    # The rheological elemnt to which this solution variable belongs 
    var_elems   = (ntuple(j -> 0, Val(length(iteration_vars)))..., Base.IteratorsMD.flatten(el_local)...);

    # extract variables that have :ε or :θ in the name (need to be summed separately in residual function)
    vars_ε      =   var_names .== :ε
    vars_θ      =   var_names .== :θ

    
    # Initial values for each of the variables
    x           = SA[ Base.IteratorsMD.flatten(values(iteration_vars))..., Base.IteratorsMD.flatten(values.(args_local))...]

    # We now know the correct variable names and index numbers for each of the variables
   # @show var_names var_elems funs_elements entry_residual x

    # Compute the residual vector
    #res = eval_residual(x, composite, args_elements, funs_elements, entry_residual, vars, vars_ε, vars_θ, var_names, var_elems)

    return  var_names, var_elems, funs_elements, entry_residual, x, args_elements_aug, vars_ε, vars_θ
end


viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
viscous1_s = LinearViscosityStress(5e19)
viscous2_s = LinearViscosityStress(1e20)

powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12) # im making up numbers
case       = :case1

composite, vars, args_solve0, args_other = if case === :case1 
    composite  = viscous1, powerlaw
    vars       = (; ε  = 1e-15) # input variables
    args_solve = (; τ  = 1e2  ) # we solve for this, initial guess
    args_other = (;) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case2
    composite  = viscous1, elastic
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2,   P = 1e6  ) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case3
    composite  = viscous1, elastic, powerlaw
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2,   P = 1e6  ) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case4
    composite  = viscous1, drucker
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2, ) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case5
    composite  = viscous1, elastic, powerlaw, drucker
    vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
    args_solve = (; τ  = 1e2,   P = 1e6,) # we solve for this, initial guess
    args_other = (; dt = 1e10) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case6
    composite  = powerlaw, powerlaw
    vars       = (; ε  = 1e-15) # input variables
    args_solve = (; τ  = 1e2) # we solve for this, initial guess
    args_other = (; ) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other

elseif case === :case7
    composite  = viscous1, viscous2, drucker, drucker, elastic, powerlaw
    vars       = (; ε  = 1e-15, θ = 0.1e-15) # input variables
    args_solve = (; τ  = 1e2, P = 10.0) # we solve for this, initial guess
    args_other = (; τ0  = 1.0, P0 = 1.0, dt = 1.0, ε  = 1e-15) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other


elseif case === :case8
    composite  = viscous1, viscous2
    composite  = viscous1, powerlaw
    
    vars       = (; ε  = 1e-15) # input variables
    args_solve = (; τ  = 1e2,) # we solve for this, initial guess
    args_other = (; dt = 1.0) # other args that may be needed, non differentiable
    composite, vars, args_solve, args_other
end


#@btime main($(composite, vars, args_solve, args_other)...)
var_names, var_elems, funs_elements, entry_residual, x, args_elements, vars_ε, vars_θ = main((composite, vars, args_solve, args_other)...)





@btime eval_residual($x, $composite, $args_elements, $funs_elements, $entry_residual, $vars, $vars_ε, $vars_θ, $var_names, $var_elems)


#eval_residual(x, composite, args_elements, funs_elements, entry_residual, vars, vars_ε, vars_θ, var_names, var_elems)

R, J = value_and_jacobian(        
    x -> eval_residual(x, composite, args_elements, funs_elements, entry_residual, vars, vars_ε, vars_θ, var_names, var_elems),
    AutoForwardDiff(), 
    x
)
@show J

#main(composite, vars, args_solve0, args_other)
# @b main($(composite, vars, args_solve, args_other)...)


