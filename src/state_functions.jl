# compute_strain_rate methods
@inline compute_strain_rate(r::LinearViscosity; τ = 0, kwargs...) = τ / (2 * r.η)
@inline compute_strain_rate(r::PowerLawViscosity; τ = 0, kwargs...) = τ^r.n / (2 * r.η)
@inline compute_strain_rate(r::Elasticity; τ = 0, τ0 = 0, dt = 0, kwargs...) = (τ - τ0) / (2 * r.G * dt)
@inline compute_strain_rate(r::IncompressibleElasticity; τ = 0, τ0 = 0, dt = 0, kwargs...) = (τ - τ0) / (2 * r.G * dt)
@inline compute_strain_rate(r::AbstractRheology; kwargs...) = 0e0 # for any other rheology that doesnt need this method
@inline function compute_strain_rate(r::DruckerPrager; τ = 0, λ = 0, P_pl = 0, kwargs...) 
    ε_pl = compute_plastic_strain_rate(r::DruckerPrager; τ_pl = τ, λ = λ, P_pl = P_pl, kwargs...)
    return ε_pl
end

# splatter wrapper
@inline compute_strain_rate(r::AbstractRheology, kwargs::NamedTuple) = compute_strain_rate(r; kwargs...)

# compute_volumetric_strain_rate methods
@inline compute_volumetric_strain_rate(r::Elasticity; P=0, P0 = 0, dt = 0, kwargs...) = (P - P0) / (r.K * dt)
@inline function compute_volumetric_strain_rate(r::DruckerPrager; τ = 0, λ = 0, P = 0, kwargs...) 
    λ * ForwardDiff.derivative(x -> compute_Q(r, τ, x), P) # perhaps this derivative needs to be hardcoded
end
@inline compute_volumetric_strain_rate(r::AbstractRheology; kwargs...) = 0e0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_volumetric_strain_rate(r::AbstractRheology, kwargs::NamedTuple) = compute_volumetric_strain_rate(r; kwargs...)

# compute_volumetric_strain_rate methods
@inline function compute_lambda(r::DruckerPrager; τ = 0, λ = 0, P = 0, kwargs...) 
    F = compute_F(r, τ, P)
    (F>0) * F - λ
end
@inline compute_lambda(r::AbstractRheology; kwargs...) = 0e0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_lambda(r::AbstractRheology, kwargs::NamedTuple) = compute_lambda(r; kwargs...)

# special plastic helper functions
function compute_F(r::DruckerPrager, τ, P) 
    F = (τ - P * sind(r.ϕ) - r.C * cosd(r.ϕ))
    F * (F > 0)
end
compute_Q(r::DruckerPrager, τ, P) = τ - P * sind(r.ψ)

# compute_stress methods
@inline compute_stress(r::AbstractRheology; kwargs...) = 0e0 # for any other rheology that doesnt need this method
@inline compute_stress(r::LinearViscosity; ε = 0, kwargs...) = ε * 2 * r.η
@inline compute_stress(r::LinearViscosityStress; ε = 0, kwargs...) = ε * 2 * r.η
@inline compute_stress(r::PowerLawViscosity; ε = 0, kwargs...) = ε^(1/r.n) * (2 * r.η)^(1/r.n)
@inline compute_stress(r::Elasticity; ε = 0, τ0 = 0, dt = 0, kwargs...) = τ0 + 2 * r.G * dt * ε
@inline compute_stress(r::IncompressibleElasticity; ε = 0, τ0 = 0, dt = 0, kwargs...) = τ0 + 2 * r.G * dt * ε
@inline compute_stress(r::DruckerPrager; τ_pl = 0, kwargs...) = τ_pl
# splatter wrapper
@inline compute_stress(r::AbstractRheology, kwargs::NamedTuple) = compute_stress(r; kwargs...)

# compute_pressure methods
@inline compute_pressure(r::Elasticity; θ = 0, P0 = 0, dt = 0, kwargs...) = P0 +  r.K * dt * θ
@inline compute_pressure(r::DruckerPrager; P_pl = 0, kwargs...) = P_pl
@inline compute_pressure(r::AbstractRheology; kwargs...) = 0e0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_pressure(r::AbstractRheology, kwargs::NamedTuple) = compute_pressure(r; kwargs...)

@inline function compute_plastic_strain_rate(r::DruckerPrager; τ_pl = 0, λ = 0, P_pl = 0, ε = 0, kwargs...) 
    λ * ForwardDiff.derivative(x -> compute_Q(r, x, P_pl), τ_pl) - ε # perhaps this derivative needs to be hardcoded
end
@inline compute_plastic_strain_rate(r::AbstractRheology; kwargs...) = 0e0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_plastic_strain_rate(r::AbstractRheology, kwargs::NamedTuple) = compute_plastic_strain_rate(r; kwargs...)

@inline function compute_volumetric_plastic_strain_rate(r::DruckerPrager; τ_pl = 0, λ = 0, P_pl = 0, θ = 0, kwargs...) 
    λ * ForwardDiff.derivative(x -> compute_Q(r, τ_pl, x), P_pl) - θ # perhaps this derivative needs to be hardcoded
end
@inline compute_volumetric_plastic_strain_rate(r::AbstractRheology; kwargs...) = 0e0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_volumetric_plastic_strain_rate(r::AbstractRheology, kwargs::NamedTuple) = compute_volumetric_plastic_strain_rate(r; kwargs...)

@inline compute_plastic_stress(r::DruckerPrager; τ_pl = 0, kwargs...) = τ_pl
@inline compute_plastic_stress(r::AbstractRheology, kwargs::NamedTuple) = compute_plastic_stress(r; kwargs...)
@inline compute_plastic_stress(r::AbstractRheology; kwargs...) = 0e0 # for any other rheology that doesnt need this method
