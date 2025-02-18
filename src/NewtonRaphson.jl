using LinearAlgebra, StaticArrays

r(τ;a=1.0,b=1.0,c=1.0,n=1.0)         = @SMatrix [a*τ[1] + b*τ[1]^n + c]
∂r∂τ(τ;a=1.0,b=1.0,n=1.0, kwargs...) = @SMatrix [a + n*b*τ[1]^(n-1)] 

# backtracking line search
function bt_line_search(x,Δx,J,args; α=1.0,ρ=0.5,c=1e-4,α_min=1e-8)
    
    a = norm(r(x .+ α*Δx; args...)) # 2.4303569995328414e-15
    b = norm(r(x; args...) .+ c*α*dot(J, Δx)) # 9.000099899999999e-15
    while norm(r(x .+ α*Δx; args...)) > 
           norm(r(x; args...) .+ c*α*dot(J, Δx))
        α *= ρ
        if α < α_min; α = α_min; break; end
    end
    return α
end

function NewtonRhapson(τ, args; maxiter=100, tol=1e-10, verbose=false)
    x         = @SMatrix [τ]
    err,iter  = 1e3, 0
    while iter < maxiter && err > tol
        iter += 1
        R    =    r(x; args...) # 9e-15
        J    = ∂r∂τ(x; args...) #  3.0000999999999997e-16       
        
        Δx   = -J\R # -30.00233325555815
        α    = bt_line_search(x, Δx, J, args)
        x    = x + α * Δx
        
        err  = norm(Δx/abs.(x))

        if verbose; println("iter: $iter, x: $x, err: $err, α = $α"); end
    end 

    return x
end 

args = (;a=1e-20, b=1e-20, n=3e0, c=-1e-15)
τ    = τ0 = 100

# NewtonRhapson(τ0, args, verbose=true)
@b NewtonRhapson($τ0, $args, verbose=false)
