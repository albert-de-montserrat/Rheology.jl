using Printf, Plots
import Statistics:mean

# Try to implement anistropic friction but resulting invariants are not objective

function main_complex_ani_vis()
    # Material parameters
    δ    = 4.0 
    # Kinematics
    pure_shear = 1
    ε̇xxd       = pure_shear*1.0
    ε̇yyd       = -ε̇xxd
    ε̇xyd       = 0#(1.0-pure_shear)*5.0 + 5.0/3
    ε̇bg        = sqrt(ε̇xxd^2 + ε̇xyd^2)
    τxx        = 0.
    τyy        = 0.
    τxy        = 0.
    P          = 1.0
    # Elasticity
    nt         = 1
    Δt         = 1e0
    G          = 1.0
    K          = 3.0  
    ηe         = G*Δt
    # Plasticity
    C          = 1250000
    fric       = 30*π/180
    dil        = 10*π/180
    a1         = 1.0
    a2         = 1.0
    a3         = 1.
    ηvp        = 1.0
    am         = 1//3*(a1+a2+a3)
    # Power law
    npwl       = 6.0001
    τbg        = 2.0
    Bpwl       = 2^npwl*ε̇bg/τbg^(npwl)
    τ_chk      = 2*Bpwl^(-1.0/npwl)*ε̇bg^(1.0/npwl)
    ηpwl       =   Bpwl^(-1.0/npwl)*ε̇bg^(1.0/npwl-1)
    τ_chk      = 2*ηpwl*ε̇bg
    Cpwl       = (2*Bpwl^(-1/npwl))^(-npwl)
    if abs(τbg - τ_chk)/τbg > 1e-6 error("Power-law breaks down") end
    # -------------------- TEST: Anisotropic power law -------------------- #
    # Arrays
    # θ = [π/3]
    θ          = LinRange( 0.0, π/2, 51 ) .-0* π/2
    τii_cart1  = zero(θ) 
    τii_cart2  = zero(θ)
    ε̇ii_rot    = zero(θ)
    τii_rot1   = zero(θ)
    τii_rot2   = zero(θ)
    η_rot      = zero(θ)
    # Loop over all orientations
    for i in eachindex(θ)
        τxx, τyy, τxy = -.0, .0, 0.0
        @show θ[i] 
        for it=1:nt
            τxx0, τyy0, τxy0 = τxx, τyy, τxy
            # Transformation matrix: towards principal plane
            Q           = [cos(θ[i]) sin(θ[i]); -sin(θ[i]) cos(θ[i])]
            ε̇_tens      = [ε̇xxd ε̇xyd; ε̇xyd ε̇yyd]
            τ0_tens     = [τxx0 τxy0; τxy0 τyy0]
            ε̇_rot       = Q*ε̇_tens*Q' 
            τ0_rot      = Q*τ0_tens*Q'

            # VISCO ELASTICITY
            ε̇_rot_eff   = [ε̇_rot[1,1]+τ0_rot[1,1]/2/ηe; ε̇_rot[2,2]+τ0_rot[2,2]/2/ηe; ε̇_rot[1,2]+τ0_rot[1,2]/2/ηe*δ]
            I2          = 0.5*(ε̇_rot_eff[1]^2 + ε̇_rot_eff[2]^2) + ε̇_rot_eff[3]^2
            ε̇ii         = sqrt(I2)
            ηpwl        =  Bpwl^(-1.0/npwl)*sqrt(I2)^(1.0/npwl-1)
            ηve         = (1.0/ηpwl + 1.0/ηe).^(-1)
            τii         = 0.
            r0          = 0.
            for iter=1:30
                τii        = 2*ηve*ε̇ii
                ε̇pwl       = Cpwl*τii^npwl
                r          = ε̇ii - τii/2/ηe - ε̇pwl
                if iter==1 r0 = r end
                ∂r∂ηve     = - ε̇ii/ηe - ε̇pwl*npwl/ηve
                ηve       -= r/∂r∂ηve
                @show (iter, abs(r)/abs(r0))
                if (abs(r)/abs(r0)<1e-9) break; end
            end
            D           = 2*ηve*[1 0 0; 0 1 0; 0 0 1.0/δ;]
            τ           = D*ε̇_rot_eff
            # ---> Independent on orientation (objective) !!!!!!
            Y2          = 0.5*(τ[1]^2 + τ[2]^2) + τ[3]^2*δ^2
            τii = sqrt(Y2); τxxt = τ[1]; τyyt = τ[2]; τzzt = -τxxt-τyyt; τxyt = τ[3]

            # PLASTICITY
            soon = 1.
            γ̇ = 0.
            τi = a1*τxxt + a2*τyyt + a3*τzzt
            F  = τii - cos(fric)*C - P*sin(fric)*am + sin(fric)*τi/3*soon
            if F>0
                coming = 2*ηve*sin(dil)*sin(fric)/9 * (a1^2 + a2^2 - a1*a3 - a2*a3) + 3/9*ηve*sin(fric)* (τxx/τii*a1 + τyy/τii*a2 - τxx/τii*a3 - τyy/τii*a3)
                γ̇    = F/(ηve + ηvp + K*Δt*sin(dil)*sin(fric)*am + coming*soon)
                τiic = τii  - γ̇*ηve
                dτi  = γ̇*ηve*(-a1*τxx - a2*τyy + a3*τxx + a3*τyy - 2/3*sin(dil)*τii*(a1^2 + a2^2 - a1*a3 - a2*a3)) /τii
                τic  = τi   + dτi
                Pc   = P    + K*Δt*sin(dil)*γ̇ 
                τxxc = τxxt - 2*ηve*γ̇*(τxxt/τii/2 + a1*sin(dil)/3*soon)
                τyyc = τyyt - 2*ηve*γ̇*(τyyt/τii/2 + a2*sin(dil)/3*soon)
                τzzc = -τxxc-τyyc
                
                τic1 = a1*τxxc + a2*τyyc + a3*τzzc
                F    = τiic - cos(fric)*C - Pc*sin(fric)*am - ηvp*γ̇ + sin(fric)*τic/3*soon
                @show (F)
                Y2   = τiic^2
                ηvep = τiic/2/ε̇ii
                ηve  = ηvep
                D    = 2*ηve*[1 0 0; 0 1 0; 0 0 1.0/δ;]
                τ    = D*ε̇_rot_eff 
                Y2   = 0.5*(τ[1]^2 + τ[2]^2) + τ[3]^2*δ^2
                τiic = sqrt(Y2)
                τic2 = a1*τ[1] + a2*τ[2] + a3*(-τ[1]-τ[2])
                @show τic, τic1, τic2 
                @show τxxt, τyyt, τzzt
                @show τxxc, τyyc, τzzc
                @show τ[1], τ[2], -τ[1]-τ[2]
                F    = τiic - cos(fric)*C - Pc*sin(fric)*am - ηvp*γ̇ + sin(fric)*τic2/3*soon
                @show (F)
            end
            τii_rot1[i] = sqrt(Y2)
            ε̇ii_rot[i]  = sqrt(I2)
            η_rot[i]    = ηve

            # Check strain rate components 
            τxx         = τ[1]
            τyy         = τ[2]
            τxy         = τ[3]
            ε̇xxd_v = Cpwl*τii^((npwl-1))*τxx
            ε̇xyd_v = Cpwl*τii^((npwl-1))*(τxy*δ)
            ε̇xxd_e = (τxx-τ0_rot[1,1])/2/ηe
            ε̇xyd_e = (τxy-τ0_rot[1,2])/2/ηe*δ
            ε̇xxd_p = γ̇*τxxt/τii/2
            ε̇xyd_p = γ̇*τxyt/τii/2*δ
            fxx    = ε̇_rot[1,1] - ε̇xxd_v - ε̇xxd_e - ε̇xxd_p
            fxy    = ε̇_rot[1,2] - ε̇xyd_v - ε̇xyd_e - ε̇xyd_p
            @show (fxx, fxy)
            # Effective viscosity: compute stress invariant from strain rate invariant
            # ---> Predicts rotated stress invariant !!!!!!
            I2          = 0.5*(ε̇_rot_eff[1]^2 + ε̇_rot_eff[2]^2) + ε̇_rot_eff[3]^2
            τii_rot2[i] = 2*ηve*sqrt(I2)
            # If one does the same by scaling the strain rate invariant by δ^2...
            # ---> Predicts Cartesian stress invariant !!!!!!
            I2          = 0.5*(ε̇_rot_eff[1]^2 + ε̇_rot_eff[2]^2) + ε̇_rot_eff[3]^2/δ^2
            τii_cart1[i] = 2*ηve*sqrt(I2)
            # Rotate stress back 
            τ_rot       = [τ[1] τ[3]; τ[3] τ[2]]
            τ           = Q'*τ_rot*Q
            # ---> Dependent on orientation (non-objective) !!!!!!
            J2          = 0.5*(τ[1,1]^2 + τ[2,2]^2) + τ[1,2]^2
            τii_cart2[i] = sqrt(J2) 
            τxx = τ[1,1]
            τyy = τ[2,2]
            τxy = τ[1,2]


            ##########################################################

            n     = θ[i] - π/2  
            nx    = cos(n)
            ny    = sin(n)
            d1    = 2*nx^2*ny^2
            d2    = nx*nx*(ny^2 - nx^2)
            ani   = 1.0 - 1.0 / δ
            ani   = 1.0 - 1.0 / δ
            A = [2.0-2.0*ani*d1 2.0*ani*d1 -2.0*ani*d2;
                 2.0*ani*d1 2.0-2.0*ani*d1 2.0*ani*d2;
                -2.0*ani*d2     2.0*ani*d2 1.0  + 2.0*ani*(d1 - 0.5);]

            E = [ε̇xxd; ε̇yyd; 2ε̇xyd]
            τ0 = [τxx0/ηe; τyy0/ηe; τxy0/ηe]
            b = A\τ0
            Eeff = E + b./[1; 1; 2]

            T = ηve*A*Eeff
            @show T
            τvec = [τxx; τyy; τxy]
            @show τvec

            # @show Eeff


            Da11  = 2.0 - 2.0*ani*d1;
            Da12  = 2.0*ani*d1;
            Da13  =-2.0*ani*d2;
            Da22  = 2.0 - 2.0*ani*d1;
            Da23  = 2.0*ani*d2;
            Da33  = 1.0  + 2.0*ani*(d1 - 0.5);
            a11   = Da33 * Da22 - Da23^2;
            a12   = Da13 * Da23 - Da33 * Da12;
            a13   = Da12 * Da23 - Da13 * Da22;
            a22   = Da33 * Da11 - Da13^2;
            a23   = Da12 * Da13 - Da11 * Da23;
            a33   = Da11 * Da22 - Da12^2;
            det   = (Da11 * a11) + (Da12 * a12) + (Da13 * a13);
            iDa11 = a11/det; 
            iDa12 = a12/det; 
            iDa13 = a13/det;
            iDa22 = a22/det; 
            iDa23 = a23/det;
            iDa33 = a33/det;
            Exx = ε̇xxd + (iDa11*τxx0 + iDa12*τyy0 + iDa13*τxy0)/ηe;
            Eyy = ε̇yyd + (iDa12*τxx0 + iDa22*τyy0 + iDa23*τxy0)/ηe;
            Exy = ε̇xyd + (iDa13*τxx0 + iDa23*τyy0 + iDa33*τxy0)/ηe/2.0;

        end
    end

    p1 = plot(title="Stress invariant", xlabel="θ", ylabel="τᵢᵢ")
    p1 = plot!(θ*180/π, τii_cart1, label="τii_cart1")
    p1 = plot!(θ*180/π, τii_cart2, label="τii_cart2", linewidth=0, marker =:cross)
    p1 = plot!(θ*180/π,  τii_rot1, label="τii_rot1")
    p1 = plot!(θ*180/π,  τii_rot2, label="τii_rot2", linewidth=0, marker =:cross)

end

main_complex_ani_vis()