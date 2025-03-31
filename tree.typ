// #import "@preview/equate:0.3.0": equate

// #show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")
#set heading(numbering: "1.1")

= Structure

```julia
struct CompositeEquation{T1, T2, F, R}
    parent::T1  # i-th element of x to be substracted
    child::T2   # i-th element of x to be added
    self::Int64 # equation number
    fn::F       # state function
    rheology::R
end
```

= Viscous -- (Viscous + Viscous)

Equations

$
  epsilon = 1/(2 eta_1) + epsilon^p
$
$
  tau = 2 eta_2 epsilon^p + 2 eta_3 epsilon^p
$

Residuals

$
  r_1(tau) = 1/(2 eta_1) + epsilon^p - epsilon = 0
$
$
  r_2(epsilon^p) = 2 eta_2 epsilon^p + 2 eta_3 epsilon^p - tau = 0
$


Then 

$
r_1 arrow cases(
  "self"   = 1,
  "parent" = (,),
  "child"  = (2,),
  "functions"  = "compute_strain_rate",
  "rheology"   = ("LinearViscous"_1, ),
)
$

$ 
r_2 arrow cases(
  "self"   = 2,
  "parent" = (1,),
  "child"  = (,),
  "functions"  = "compute_stress",
  "rheology"   = ("LinearViscous"_1, "LinearViscous"_2, ),
)
$

= Viscous -- (Viscous + Viscous) -- (Viscous + Viscous)

Equations

$
  epsilon = 1/(2 eta_1) + epsilon^p_1 + epsilon^p_2
$
$
  tau = 2 eta_2 epsilon^p_1 + 2 eta_3 epsilon^p_1
$
$
  tau = 2 eta_4 epsilon^p_2 + 2 eta_4 epsilon^p_2
$

Residuals

$
  r_1(tau) = 1/(2 eta_1) + epsilon^p_1 + epsilon^p_2 - epsilon = 0
$
$
  r_2(epsilon^p_1) = 2 eta_2 epsilon^p_1 + 2 eta_3 epsilon^p_1 - tau = 0
$
$
  r_3(epsilon^p_2) = 2 eta_4 epsilon^p_2 + 2 eta_5 epsilon^p_2 - tau = 0
$


Then 

$
r_1 arrow cases(
  "self"   = 1,
  "parent" = (,),
  "child"  = (2,3,),
  "functions"  = "compute_strain_rate",
  "rheology"   = ("LinearViscous"_1, ),
)
$

$ 
r_2 arrow cases(
  "self"   = 2,
  "parent" = (1,),
  "child"  = (,),
  "functions"  = "compute_stress",
  "rheology"   = ("LinearViscous"_1, "LinearViscous"_2, ),
)
$

$ 
r_3 arrow cases(
  "self"   = 3,
  "parent" = (1,),
  "child"  = (,),
  "functions"  = "compute_stress",
  "rheology"   = ("LinearViscous"_1, "LinearViscous"_2, ),
)
$