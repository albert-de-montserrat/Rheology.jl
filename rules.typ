// #import "@preview/equate:0.3.0": equate

// #show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")

= Rheology types

== Linear viscous

Strain rate:
$
  dot(epsilon) = 1 / (2 eta)tau
$

Stress:
$
  tau = 2 eta dot(epsilon)
$

== Power law

Strain rate:
$
  dot(epsilon) = 1 / (2 eta_o)tau^n
$

Stress:
$
  tau = (2 eta_o dot(epsilon))^(1/n)
$

== Linear elasticity

Strain rate:
$
  dot(epsilon) = 1 / (2 G)  (partial tau) / (partial t) approx 1 / (2 G)  (tau - tau^o) / (Delta t)
$

Stress:
$
tau = 2 G Delta t dot(epsilon) + tau^o
$

Volumetric strain rate:
$
  theta = 1 / K (P-P^o) / (Delta t)
$

Pressure:
$
  P = theta K Delta t + P^o  
$

== Drucker-Prager
Plastic strain rate:
$
  dot(epsilon)_("ij") = dot(lambda) (partial Q) / (partial tau_("ij"))
$

where typically 
$
  Q = tau_("II") - P sin(psi)
$

and $dot(lambda) > 0$ only if $F>0$, where 
$
  F = tau_("II") - P sin(phi) - C cos(phi)
$

Volumetric plastic strain rate
$
  theta = dot(lambda) (partial Q) / (partial P)
$

= Exclusively series elements

Strain rate is the sum of the strain rate of all the components
$
  dot(epsilon) = sum_i^n dot(epsilon)_i
$

and stress is equal for all components
$
  tau = tau_1 = dots.h = tau_n
$

== Example 1: 

Components: linear viscous + power law + linear elasticity

$
  dot(epsilon) = dot(epsilon)^("viscous") + dot(epsilon)^("power law") + dot(epsilon)^("elasticity") =
  1 / (2 eta)tau + 1 / (2 eta_o)tau^n + 1 / (2 G)  (tau - tau^o) / (Delta t)
$<ex1:strain>

and
$
  theta = theta^("viscous") + theta^("power law") + theta^("elasticity") =
  0 + 0 + 1 / K (P-P^o) / (Delta t)
$<ex1:vol>

We need to solve for both $epsilon$ and $P$. To build the Newton-Raphson solver we need the residual functions // @ref{ex1:strain} and @ref{ex1:vol}

$ r = mat(
  r(tau);
  r(P);
) = 
mat(
  1 / (2 eta)tau + 1 / (2 eta_o)tau^n + 1 / (2 G)  (tau - tau^o) / (Delta t) - dot(epsilon);
  1 / K (P-P^o) / (Delta t) - theta;
) 
$

and the corresponding Jacobian matrix

$ 
J = mat(
  (partial r(tau)) / (partial tau), (partial r(tau)) / (partial P);
  (partial r(P)) / (partial tau), (partial r(P)) / (partial P);
) = mat(
  sum_i^n (partial r_i (tau)) / (partial tau), sum_i^n (partial r_i (tau)) / (partial P);
  sum_i^n (partial r_i (P)) / (partial tau)  , sum_i^n (partial r_i (P)) / (partial P);
) = 
mat(
  1 / (2 eta) + n / (2 eta_o)tau^(n-1) + 1 / (2 G Delta t), 0;
  0, 1 / (K Delta t);
) 
$

= Exclusively parallel elements

Strain rate  is equal for all components 
$
  dot(epsilon) = dot(epsilon)_1 = dots.h = dot(epsilon)_n
$

and stress is the sum of the strain rate of all the components
$
  tau = sum_i^n tau_i
$

== Example 2: Solving for stress and pressure

Components: linear viscous + power law + linear elasticity

$
  dot(epsilon) = dot(epsilon)^("viscous") = dot(epsilon)^("power law") = dot(epsilon)^("elasticity")
$

$
  dot(epsilon)^("viscous")    = dot(epsilon) = 1 / (2 eta)tau^("viscous") = 1 / (2 eta)tau_1
$
$
   dot(epsilon)^("power law")  = dot(epsilon) = 1 / (2 eta_o)(tau^("power law"))^n = 1 / (2 eta_o)tau_2^n
$ 
$
  dot(epsilon)^("elasticity") = dot(epsilon) = 1 / (2 G)  (tau^("elasticity") - tau^o) / (Delta t) = 1 / (2 G) (tau_3 - tau^o) / (Delta t) \
$
and $tau = tau_1 + tau_2 + tau_3$

$
  x = mat(
    tau;
    tau_1;
    tau_2;
    tau_3;
  )
$
$
  r_1 = mat(
    r(tau);
    r(tau_1);
    r(tau_2);
    r(tau_3);
  ) = 
  mat(
    tau - tau_1 - tau_2 - tau_3;
    1 / (2 eta)tau_1 - dot(epsilon);
    1 / (2 eta_o)tau_2^n - dot(epsilon);
    1 / (2 G) (tau_3 - tau^o) / (Delta t) - dot(epsilon);
  )
$
and 
$
  J_1 = mat(
    1, 0, 0, 0;
    0, 1/(2 eta), 0, 0;
    0, 0, n/(2 eta_o)tau_2^n, 0;
    0, 0, 0, 1/(2 G Delta t);
  )
$

and we do the same for volumetric strain rate

$
  theta = theta^("elasticity") = 1 / (K) (P_3 - P^o) / (Delta t)
$

with 
$
  P = P_1 + P_2 + P_3 = P_3
$

Then
$
  x = mat(
    theta;
    theta_1;
  )
$
$
  r_2 = mat(
    r(P);
    r(P_3);
  ) = 
  mat(
    P - P_3;
    1 / (K) (P_3 - P^o) / (Delta t) - theta;
  )
$
and
$
  J_2 = mat(
    1, 0;
    0, 1/(K Delta t);
  )
$

So that we have
$
  r = mat(
    r_1;
    r_2;
  )
$
$
  J = mat(
    J_1, 0;
    0, J_2;
  )
$
Note: the off-diagonal terms of $J$ may not be zero if strain rate depends on P.

=== What do we need to generate this code?
For one simple parallel element we need:

  + Get all the unique state functions and variables
  + For a state variable $Psi$ generate the residual function $r(Psi) = Psi - sum_i^n Psi_i$
  + Generate the local $Psi_i$ state functions and their residuals functions $r_i=r(Psi_i)$. 
    + About residual functions: we just need *NOT* to flatten the state functions for the parallel element.
    + About local state variables: we "dont really care" about them, but we can store them in a _Tuple_ of NamedTuple matching the state functions, so that we can do pattern matching using the _kwargs_ trick. 


