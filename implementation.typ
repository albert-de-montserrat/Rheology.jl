// #import "@preview/equate:0.3.0": equate

// #show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")
#set heading(numbering: "1.1")

= Problem
$
  epsilon = 1 / (2 eta) tau + 1 / (2 G Delta t) tau + epsilon^("pl")
$

$
  theta = 1/(K Delta t)(P-P^"o")
$

$
  tau = (2 eta epsilon^("pl"))^(1/n)
$

$
  x = mat(
    tau;
    P;
    epsilon^("pl");
  )
$
$
  r = mat(
    -epsilon + 1 / (2 eta) tau + 1 / (2 G Delta t) tau + epsilon^("pl");
    -theta + 1/(K Delta t)(P-P^"o");
    -tau + (2 eta epsilon^("pl"))^(1/n);
  )
$

$
  J = mat(
    1 / (2 eta) + 1 / (2 G Delta t), 0,                                                        1;
    0,                       P^"o"/(K Delta t),                                        0;
   -1,                       0,                 ((2 eta)^(1/n) epsilon^"pl"^(1/n-1)) / n;
  )
$


= Implementation

Need to tear the residual into several pieces

+ $
  r_1 = mat(
    -epsilon ;
    -theta   ;
    -tau^"pl";
  )
$ where $"vars" =(; epsilon, theta)$ are input variables, i.e. immutable; and, $tau^"pl"$ is an unknown.
+ $
  r_2 = mat(
    1 / (2 eta) tau + 1 / (2 G Delta t) tau;
    1/(K Delta t)(P-P^"o");
    0;
  ) = 
  mat(
    sum_i^n_1  f(tau);
    sum_i^n_2  f(P);
    0;
  ) =
  mat(
    sum_i^n_1  f_i(x_1);
    sum_i^n_2  f_i(x_2);
    0;
  ) 
$
+ $
  r_3 = mat(
    epsilon^("pl");
    0;
    (2 eta epsilon^("pl"))^(1/n);
  )
  = mat(
    f(x_3);
    0;
    h(x_3);
  )
$ This is the simple case where there is only one local element defined by compute_stress. If we had e.g. 2 elements, then $f(x_3) arrow g(x_3, x_4) = f(x_3) + f(x_4)$. Therefore the array mapping the variables in $x$ to the function arguments would be 
$
  "inds" = mat(
  (3,4);
  (0,);
  (3,);
  (4,);
  ) 
$
