using GLMakie

include("mwe_parallel.jl")



viscous  = LinearViscosity(5e19)
powerlaw = PowerLawViscosity(5e19, 3)



τ =  10.0.^(-4:.1:4)


#composite = viscous, powerlaw
#composite = powerlaw,
#composite = viscous,

function compute_strain_rate_parallel(τ, composite)
    ε = zero(τ)

    for (I,τ_loc) in enumerate(τ)
        input_vars = (; τ = τ_loc)
        args_guess =  (; ε = 1e-35)

        output = main_parallel(input_vars, composite, args_guess; verbose = false)

        ε[I] = output.ε
    end

    return ε
end


function compute_strain_rate_series(ε, composite)
    τ = zero(ε)

    for (I,ε_loc) in enumerate(ε)
        input_vars = (; ε = ε_loc)
        args_guess =  (; τ = 1.0)

        output = main_series(input_vars, composite, args_guess; verbose = false)

        τ[I] = output.τ
    end

    return τ
end

args_guess = (; ε = 1e-15) # we solve for this, initial guess
input_vars = (; τ = 1e2, ) # input variables



composite = (powerlaw, viscous)

ε_linear     = compute_strain_rate_parallel(τ, (viscous,  ))
ε_powerlaw   = compute_strain_rate_parallel(τ, (powerlaw, ))
ε_composite_parallel  = compute_strain_rate_parallel(τ, composite)
τ_composite_series  = compute_strain_rate_series(ε_linear, composite)


input_vars = (; τ = 0.01)
args_guess =  (; ε = 1e-27)
output = main_parallel(input_vars, composite, args_guess; verbose = false)



fig = Figure()
fntsize = 40
title_str = "composite linear + powerlaw viscous rheologies"

ax = Axis(fig[1, 1], xlabel = L"$\dot{\varepsilon} \textrm{ [s^{-1}]}$", ylabel = L"$\tau \textrm{ [Pa]}$", 
            title = title_str, xscale=log10, yscale=log10, xlabelsize = fntsize, ylabelsize = fntsize, titlesize = fntsize,
            xticklabelsize=fntsize, yticklabelsize=fntsize)
li1 = lines!(ax, ε_viscous,   τ, color=:blue, linewidth=2,  label="linear")
li2 = lines!(ax, ε_powerlaw,  τ, color=:red, linewidth=2,   label="powerlaw")
li3 = lines!(ax, ε_composite_parallel, τ, color=:black,      linestyle=:dot, linewidth=4, label="composite parallel")
li4 = lines!(ax, ε_linear, τ_composite_series, color=:green, linestyle=:dot, linewidth=4, label="composite series")

le = axislegend(ax,  position = :rb, labelsize=fntsize)

display(fig)



