
function compute_stress(r::AbstractRheology; kwargs...)
    return 0 # for any other rheology that doesn't need this method
end

function compute_stress(r::LinearViscosity; ε = 0, kwargs...)
    return ε * 2 * r.η
end

function compute_stress(r::LinearViscosityStress; ε = 0, kwargs...)
    return ε * 2 * r.η
end

function compute_stress(r::PowerLawViscosity; ε = 0, kwargs...)
    return ε^(1 / r.n) * (2 * r.η)^(1 / r.n)
end

function compute_stress(r::Elasticity; ε = 0, τ0 = 0, dt = 0, kwargs...)
    return τ0 + 2 * r.G * dt * ε
end

function compute_stress(r::IncompressibleElasticity; ε = 0, τ0 = 0, dt = 0, kwargs...)
    return τ0 + 2 * r.G * dt * ε
end

function compute_stress(r::DruckerPrager; τ_pl = 0, kwargs...)
    return τ_pl
end

@inline function compute_stress(r::NonNewtonianCreep{_T}; ε = 0, f=1, T = 1, P = 0, d = 1, kwargs...) where _T
    (; n, p, A, E, V, R) = r
    n_inv = inv(n)
    τ = A^-n_inv * ε^n_inv * f^(-r * n_inv) * d^(-p * n_inv) * exp((E + P * V) / (n * R * T)) / FT
    return τ
end

@inline function compute_stress(r::GrainBoundarySliding; τ = 0, T = 1, P = 0, d = 1, kwargs...)
    (; n, p, A, E, V, R) = r
    β = isone(d) ? one(_T) : d^p
    θ = isone(n) ? τ : τ^n
    ε = A * θ * β * exp(-(E + P * V) / (R * T))
    return ε
end

# splatter wrapper
@inline compute_stress(r::AbstractRheology, kwargs::NamedTuple) = compute_stress(r; kwargs...)