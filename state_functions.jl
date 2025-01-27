# compute_shear_strain methods
@inline compute_shear_strain(r::LinearViscosity; τ = 0, kwargs...) = τ / (2 * r.η)
@inline compute_shear_strain(r::PowerLawViscosity; τ = 0, kwargs...) = τ^r.n / (2 * r.η)
@inline compute_shear_strain(r::Elasticity; τ = 0, τ0 = 0, dt =Inf, kwargs...) = (τ - τ0) / (2 * r.G * dt)
@inline compute_shear_strain(r::AbstractRheology; kwargs...) = 0 # for any other rheology that doesnt need this method
@inline function compute_shear_strain(r::DruckerPrager; τ = 0, λ = 0, P = 0, kwargs...) 
    λ * ForwardDiff.derivative(x -> compute_F(r, x, P), τ) # perhaps this derivative needs to be hardcoded
end
# splatter wrapper
@inline compute_shear_strain(r::AbstractRheology, kwargs::NamedTuple) = compute_shear_strain(r; kwargs...)

# compute_volumetric_strain methods
@inline compute_volumetric_strain(r::Elasticity; P=0, P0 = 0, dt =Inf, kwargs...) = (P - P0) / (r.K * dt)
@inline function compute_volumetric_strain(r::DruckerPrager; τ = 0, λ = 0, P = 0, kwargs...) 
    λ * ForwardDiff.derivative(x -> compute_Q(r, τ, x), P) # perhaps this derivative needs to be hardcoded
end
@inline compute_volumetric_strain(r::AbstractRheology; kwargs...) = 0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_volumetric_strain(r::AbstractRheology, kwargs::NamedTuple) = compute_volumetric_strain(r; kwargs...)


# compute_volumetric_strain methods
@inline function compute_lambda(r::DruckerPrager; τ = 0, λ = 0, P = 0, kwargs...) 
    F = compute_F(r, τ, P)
    (F > 0) * λ
end

@inline compute_lambda(r::AbstractRheology; kwargs...) = 0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_lambda(r::AbstractRheology, kwargs::NamedTuple) = compute_lambda(r; kwargs...)


# special plastic helper functions
compute_F(r::DruckerPrager, τ, P) = τ - P * sind(r.ϕ) - r.C * cosd(r.ϕ)
compute_Q(r::DruckerPrager, τ, P) = τ - P * sind(r.ψ)


# compute_stress methods
@inline compute_stress(r::AbstractRheology; kwargs...) = 0 # for any other rheology that doesnt need this method
@inline compute_stress(r::LinearViscosity; ε = 0, kwargs...) = ε * 2 * r.η
@inline compute_stress(r::PowerLawViscosity; ε = 0, kwargs...) = ε^(1/r.n) * (2 * r.η)^(1/r.n)
@inline compute_stress(r::Elasticity; ε = 0, τ0 = 0, dt =Inf, kwargs...) = τ0 + 2 * r.G * dt * ε
# splatter wrapper
@inline compute_stress(r::AbstractRheology, kwargs::NamedTuple) = compute_stress(r; kwargs...)

# compute_pressure methods
@inline compute_pressure(r::Elasticity; θ = 0, P0 = 0, dt =Inf, kwargs...) = P0 +  r.K * dt * θ
# @inline function compute_pressure(r::DruckerPrager; τ = 0, λ = 0, P = 0, kwargs...) 
#     λ * ForwardDiff.derivative(x -> compute_Q(r, τ, x), P) # perhaps this derivative needs to be hardcoded
# end
@inline compute_pressure(r::AbstractRheology; kwargs...) = 0 # for any other rheology that doesnt need this method
# splatter wrapper
@inline compute_pressure(r::AbstractRheology, kwargs::NamedTuple) = compute_pressure(r; kwargs...)

