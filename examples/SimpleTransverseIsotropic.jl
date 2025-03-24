using Plots, LaTeXStrings, Printf, LinearAlgebra
import Statistics:mean

function main_simple_ani_vis()
    # Material parameters
    δ          = 2.0 # anisotropy strength
    ηv         = 1.0 # normal viscosity (``pure shear'' viscosity)
    # Kinematics
    pure_shear   = 1
    ε̇xxd         = pure_shear*.5
    ε̇yyd         = -ε̇xxd
    ε̇xyd         = ((1.0 - pure_shear)*5.0)
    # Arrays
    θ            = LinRange( 0.0, π, 51 ) 
    τii_cart     = zero(θ)
    τii_cart_MD7 = zero(θ)
    τii_rot1     = zero(θ)
    τii_rot2     = zero(θ)
    τxx_cart     = zero(θ)
    τxy_cart     = zero(θ)
    τxx_rot      = zero(θ)
    τxy_rot      = zero(θ)
    τ_cart       = zero(θ)
    τii_cart2    = zero(θ)
    τxx_cart2    = zero(θ)
    τyy_cart2    = zero(θ)
    τxy_cart2    = zero(θ)
    # Loop over all orientations 
    for i in eachindex(θ)

        ##########################################################
        # ACTIVITY 1: Working directly in the material coordinates
        ##########################################################

        # STEP 1: Transformation matrix: towards principal plane
        Q           = [cos(θ[i]) sin(θ[i]); -sin(θ[i]) cos(θ[i])]
        ε̇_tens      = [ε̇xxd ε̇xyd; ε̇xyd ε̇yyd]
        ε̇_rot       = Q*ε̇_tens*Q'  
        # STEP 2: Transverse isotropic viscous model 
        ε           = [ε̇_rot[1,1]; ε̇_rot[2,2]; ε̇_rot[1,2]]
        D           = 2*ηv*[1 0 0; 0 1 0; 0 0 1.0/δ;]
        τ           = D*ε
        # STEP 3: Invariants  
        # ACHTUNG: this invariant is not the regular one
        # It's a ellipse whose flattening is proportional to the anitropy strength
        # You can refer to Flectcher (2005) or any model derived from Hill's plasticity model
        Y2          = 0.5*(τ[1]^2 + τ[2]^2) + τ[3]^2*δ^2
        τii_rot1[i] = sqrt(Y2)
        # CHECK 1: Check that additive strain rate decomposition is repected
        # For more more complex models, standard LI will is still needed
        τxx         = τ[1]
        τxy         = τ[3]
        ε̇xxd_chk    = τxx/2/ηv
        ε̇xyd_chk    = τxy/2/(ηv/δ)
        @show (ε̇_rot[1,1]- ε̇xxd_chk, ε̇_rot[1,2]- ε̇xyd_chk)
        τxx_rot[i]  = τxx
        τxy_rot[i]  = τxy
        # CHECK 2: Effective viscosity: compute stress invariant from strain rate invariant
        I2          = 0.5*(ε[1]^2 + ε[2]^2) + ε[3]^2
        τii_rot2[i] = 2*ηv*sqrt(I2)
        # CHECK 3: if one does the same by scaling the strain rate invariant by δ...
        # ... then the Cartesian value is recovered  
        I2          = 0.5*(ε[1]^2 + ε[2]^2) + ε[3]^2/δ^2
        τii_cart2[i] = 2*ηv*sqrt(I2)
        # STEP 3: Transform stress back and compute ``invaraint'' in Cartesian coordinate
        # J2 is no more invariant becaus it depends on the angle and strength of anisotropy :D  
        τ_rot       = [τ[1] τ[3]; τ[3] τ[2]]
        τ           = Q'*τ_rot*Q
        J2          = 0.5*(τ[1,1]^2 + τ[2,2]^2) + τ[1,2]^2
        τii_cart[i] = sqrt(J2)  
        τxx_cart[i] = τ[1,1]
        τxy_cart[i] = τ[1,2]

        #######################################################################
        # ACTIVITY 2: Working in Cartsian coordinates with the viscosity tensor
        #######################################################################

        # Prediction of stress using viscosity tensor in Cartesian plane

        # STEP 1: Construct viscosity tensor
        two      = 2.
        n        = θ[i] - π/2 # layer is oriented at 90 degrees from the director
        nx       = cos(n); ny = sin(n)
        N        = [nx; ny]
        d0       = 2*N[1]^2*N[2]^2
        d1       = N[1]*N[2] * (-N[1]^2 + N[2]^2)
        C_ANI    = [-d0 d0 -two*d1; d0 -d0 two*d1; -d1 d1 -two*(1/2-d0)]      ###### !!! - sign in D33 -a0
        C_ISO    = [1 0 0; 0 1 0; 0 0 two*1//2]
        ani      = 1. - 1.  ./ δ
        Dani     = C_ISO .+ ani*C_ANI # Achtung: factor 2 removed from Dani

        # STEP 2: Effective strain rate (may contaain visco-elastic contributions)
        ε        = [ε̇xxd; ε̇yyd; ε̇xyd]
        # τ0       = [τxx0; τyy0; τxy0]
        ε_eff    = ε #.+ inv(Dani) * τ0 ./(2*ηe)

        # STEP 3: Predict stress vector
        τ_MD7_v1 = 2*ηv*Dani*ε_eff

        # STEP 4: Compute the non-invariant J2
        τii_cart_MD7[i]          = sqrt(0.5*(τ_MD7_v1[1]^2 + τ_MD7_v1[2]^2) + τ_MD7_v1[3]^2)

        # Check with analytics
        eta_ve, delta, theta = ηv, δ, θ[i]
        Exx, Eyy, Exy = ε̇xxd, ε̇yyd, ε̇xyd
        τ_cart[i] = 2.0 * eta_ve .* sqrt(0.25 * Exx .^ 2 .* delta .^ 2 .* cos(2 * theta) .^ 2 + 0.25 * Exx .^ 2 .* delta .^ 2 - 0.25 * Exx .^ 2 .* cos(2 * theta) .^ 2 + 0.25 * Exx .^ 2 + 0.5 * Exx .* Exy .* delta .^ 2 .* sin(4 * theta) - Exx .* Exy .* sin(4 * theta) / 2 - 0.5 * Exx .* Eyy .* delta .^ 2 .* cos(2 * theta) .^ 2 + 0.5 * Exx .* Eyy .* delta .^ 2 + 0.5 * Exx .* Eyy .* cos(2 * theta) .^ 2 - 0.5 * Exx .* Eyy - Exy .^ 2 .* delta .^ 2 .* cos(2 * theta) .^ 2 + Exy .^ 2 .* delta .^ 2 + Exy .^ 2 .* cos(2 * theta) .^ 2 - 0.5 * Exy .* Eyy .* delta .^ 2 .* sin(4 * theta) + Exy .* Eyy .* sin(4 * theta) / 2 + 0.25 * Eyy .^ 2 .* delta .^ 2 .* cos(2 * theta) .^ 2 + 0.25 * Eyy .^ 2 .* delta .^ 2 - 0.25 * Eyy .^ 2 .* cos(2 * theta) .^ 2 + 0.25 * Eyy .^ 2) ./ delta

        τv           = τii_rot1[i]
        τxx_cart2[i] = τv   * cos(2*θ[i]) 
        τyy_cart2[i] =-τv   * cos(2*θ[i])
        τxy_cart2[i] = τv/δ * sin(2*θ[i])

        ev = eigvecs(τ)
        @show atand(ev[2,1]/ev[1,1])

    end

    p1=plot(title=L"A) $\tau_{II}$ and stress invariant $\tau_{II}'$", xlabel=L"$\theta$ [$^\circ$]", ylabel=L"$\tau_{II}$, $\tau_{II}'$  [-]")
    
    # In cartesian coordinates
    plot!( θ*180/π, τii_cart, label=L"$\tau_{II}$  $(\delta$=2)$" )
    scatter!(θ*180/π, τii_cart_MD7 )
    tii = 1.0*sqrt.((δ^2-1)*cos.(2*θ).^2 .+ 1)/δ
    scatter!(θ*180/π, tii, marker=:xcross )
    
    # In material coordinates
    plot!(θ*180/π, τii_rot1, label=L"$\tau_{II}'$ $(\delta$=2)$")
    scatter!(θ[1:5:end]*180/π, τii_rot2[1:5:end], label=L"$\tau_{II}$  $(\delta$=1)$" )

    p2=plot(title=L"$$B) Flow enveloppe ($\delta = 2$)", xlabel=L"$\tau_{xx}'$ [-]", ylabel=L"$\tau_{xy}'$ [-]", aspect_ratio=1)
    plot!(LinRange(0,1,10), zeros(10), color=:black)
    plot!(zeros(10), LinRange(0,0.5,10), color=:black)
    a = LinRange(0, 2π, 100)
    txx, txy = 1*cos.(a), 1*sin.(a)/δ
    plot!(τxx_rot, τxy_rot, label=L"$\delta$=2$" )
    scatter!(txx, txy, marker=:xcross )
    
    display(plot(p1, p2, layout=(2,1)))

end

main_simple_ani_vis()