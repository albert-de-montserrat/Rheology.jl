using LinearAlgebra

r(τ;a=1.0,b=1.0,c=1.0,n=1.0)         = a*τ +   b*τ^n  + c
∂r∂τ(τ;a=1.0,b=1.0,n=1.0, kwargs...) = a   + n*b*τ^(n-1) 

# backtracking line search
function bt_line_search(x,Δx,dRdτ,args; α=1.0,ρ=0.5,c=1e-4,α_min=1e-8)
    while abs(r(x + α*Δx; args...)) > 
           abs(r(x; args...) + c*α*dot(dRdτ, Δx))
        α *= ρ
        if α < α_min; α = α_min; break; end
    end
    return α
end

function NewtonRhapson(τ, args; maxiter=100, tol=1e-10, verbose=false)
    x         = τ
    err,iter  = 1e3, 0
    while iter < maxiter && err > tol
        iter += 1
        R    =    r(x; args...) 
        dRdτ = ∂r∂τ(x; args...)
        Δx   = -inv(dRdτ)*R
        α    = bt_line_search(x, Δx, dRdτ, args)
        x    = x + α * Δx
        err  = abs(Δx)/abs(x)

        if verbose; println("iter: $iter, x: $x, err: $err, α = $α"); end
    end 

    return x
end 

args = (;a=1e-20, b=1e-20, n=3, c=-1e-15)
τ0   = 100
τ    = NewtonRhapson(τ0, args, verbose=true)