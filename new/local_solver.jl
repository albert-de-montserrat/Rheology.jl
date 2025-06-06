

function bt_line_search(Δx, J, x, r, composite, vars, others; α = 1.0, ρ = 0.5, c = 1.0e-4, α_min = 1.0e-8)

    perturbed_x = @. x - α * Δx
    perturbed_r = compute_residual(composite, perturbed_x, vars, others)

    while sqrt(sum(perturbed_r .^ 2)) > sqrt(sum((r + (c * α * (J * Δx))) .^ 2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_x = @. x - α * Δx
        perturbed_r = compute_residual(composite, perturbed_x, vars, others)
    end
    return α
end

"""
    x = solve(c::AbstractCompositeModel, x::SVector, vars, others; tol = 1.0e-9, itermax = 1e4, verbose=true)

Solve the system of equations defined by the composite model `c` using a Newton-Raphson method.
"""
function solve(c::AbstractCompositeModel, x::SVector, vars, others; tol = 1.0e-9, itermax = 1e4, verbose=true)

    it = 0
    er = Inf
    local α
    while er > tol
        it += 1
        r = compute_residual(c, x, vars, others)
        J = ForwardDiff.jacobian(y -> compute_residual(c, y, vars, others), x)
        Δx = J \ r
        α = bt_line_search(Δx, J, x, r, c, vars, others)
        x -= α .* Δx

        er = norm(iszero(xᵢ) ? 0.0e0 : Δxᵢ / abs(xᵢ) for (Δxᵢ, xᵢ) in zip(Δx, x)) # norm(r)

        it > itermax && break
    end
    if verbose
        println("Iterations: $it, Error: $er, α = $α")
    end
    return x
end

