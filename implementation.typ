// #import "@preview/equate:0.3.0": equate

// #show: equate.with(breakable: true, sub-numbering: true)
//#set math.equation(numberi: "(1.1)")
#set heading(numbering: "1.1")

= Problem 1: series
$
  epsilon = 1 / (2 eta) tau + 1 / (2 G Delta t) (tau - tau^("o")) + epsilon^("pl")
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
    -epsilon + 1 / (2 eta) tau + 1 / (2 G Delta t) (tau - tau^("o")) + epsilon^("pl");
    -theta + 1/(K Delta t)(P-P^"o");
    -tau + (2 eta epsilon^("pl"))^(1/n);
  )
$

$
  J = mat(
    1 / (2 eta) + 1 / (2 G Delta t), 0,                                         1;
    0,                       1/(K Delta t),                                        0;
   -1,                       0,                 ((2 eta)^(1/n) epsilon^"pl"^(1/n-1)) / n;
  )
$

== Implementation

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


= Problem 2: series(visco)-parallel(visco-power law)

Total strain rate $epsilon$:
$
  epsilon = epsilon^"viscous" + epsilon^"parallel" = 1 / (2 eta_1) tau + epsilon^("p")
$
Total stress
$
  tau = tau_2 + tau_3 = 2 eta _2 epsilon^("p") + (2 eta _3 epsilon^("p"))^(1/n)
$
Strain rates of the individual elements in parallel
$
  epsilon^("p") = 1 / (2 eta_2) tau_2 arrow tau_2 = 2 eta _2 epsilon^("p")
$
$
  epsilon^("p") = 1 / (2 eta_3) tau_3^n arrow tau_3 = (2 eta _3 epsilon^("p"))^(1/n)
$

Then 
$
  x = mat(
    tau;
    epsilon^("p");
  )
$

$
  r = mat(
    r(tau);
    r(epsilon^("p"));
  ) = mat(
    -epsilon + 1 / (2 eta_1) tau + epsilon^("p");
    -tau + 2 eta _2 epsilon^("p") + (2 eta_3 epsilon^("p"))^(1/n);
  )
$
$
  J = mat(
    1 / (2 eta_1), 1;
    -1, 2 eta _2 + (2 eta_3)^(1/n)/n (epsilon^"p")^(1/n-1);
  )
$
// $
//   x = mat(
//     tau;
//     epsilon^("p");
//   )
// $

// $
//   r = mat(
//     r(tau);
//     r(epsilon^("p"));
//   ) = mat(
//     -tau + 2 eta _2 epsilon^("p") + (2 eta _2 epsilon^("p"))^n;
//     -epsilon + 1 / (2 eta_1) tau + epsilon^("p");
//   )
// $
// $
//   J = mat(
//     2 eta _2 + (2 eta _2)^(n) n epsilon^"p"^(n-1), -1;
//     1, 1 / (2 eta_1),;
//   )
// $

== What do we need

+ Compute_strain_rate where we sum for all the individual series elements, and add one unknown per parallel element $
  r(epsilon^"p") = sum_i^N^"series" epsilon_i^"series" + epsilon_1^"parallel" + dots.h + epsilon_(N^"parallel")^"parallel"
$ which is better split into $
  r(epsilon^"p") = r_1(epsilon^"series") + r_2(epsilon^"parallel")
$ where $r_1$ can be pre-computed as it is constant thrughout the Newton iterations.

+ Mapping that indicates where the required $epsilon_(i^"parallel")^"parallel"$ are needed in the system of equations
$
  "local variable to system of equations mapping" =  mat(
    (2,);
    (2,);
  )
$ If we had something like
$
  x = mat(
    tau;
    epsilon_1^("p");
    epsilon_2^("p");
  )
$ Then
$
  "local variable to system of equations mapping" =  mat(
    (2,3);
    (2,);
    (3,);
  )
$



$
  r = mat(
    mat(
      "r"_"global equations";
      dots.v;
      "r"_"local series equations";
    );
    mat(
      "r"_"global equations parallel"_1;
      dots.v;
      "r"_"local parallel equations parallel"_1;
    );
    dots.v;
    mat(
      "r"_"global equations parallel"_n;
      dots.v;
      "r"_"local parallel equations parallel"_n;
    )
  )
$


= Problem 3: series(visco)-parallel(series(visco - power law) - power law)

Total strain rate of first series element $epsilon$:
$
  epsilon = epsilon^"viscous" + epsilon^"parallel" = 1 / (2 eta_1) tau + epsilon^("p")
$
Total stress of parallel element
$
  tau = tau_2 + tau_3 = tau_2 + (2 eta _2 epsilon^("p"))^(1/n)
$

Total strain rate of second series element $epsilon^"p"$:
$
  epsilon^"p" = epsilon^"viscous2" + epsilon^"powerlaw2" = 1 / (2 eta_3) tau_2 +  1 / (2 eta_4) tau_2^n
$

Then 
$
  x = mat(
    tau;
    epsilon^("p");
    tau_2
  )
$

$
  r = mat(
    r(tau);
    r(epsilon^("p"));
    r(tau_2);
  ) = mat(
    -epsilon + 1 / (2 eta_1) tau + epsilon^("p");
    -tau + tau_2 + (2 eta_2 epsilon^("p"))^(1/n);
    -epsilon^"p" + 1 / (2 eta_3) tau_2 +  1 / (2 eta_4) tau_2^"n";
  )
$

$
  J = mat(
    1 / (2 eta_1), 1, 0;
    -1, 2 eta _2 + ((2 eta_2)^(1/n) /n) (epsilon^"p")^(1/n-1), 1;
    0,-1,1 / (2 eta_3) + n / (2 eta_4) tau_2^(n-1);
  )
$

// $
//   x = mat(
//     tau;
//     epsilon^("p");
//     tau_1;
//     tau_2;
//   )
// $

// $
//   r = mat(
//     r(epsilon^("p"));
//     r(tau);
//     r(tau_2);
//     r(tau_3);
//   ) = mat(
//     -epsilon + 1 / (2 eta_1) tau + epsilon^("p");
//     -tau + 2 eta _2 epsilon^("p") + (2 eta _2 epsilon^("p"))^n;
//     -tau_2 + 2 eta _2 epsilon^("p");
//     -tau_3 + (2 eta _3 epsilon^("p"))^n;
//   )
// $
// $
//   J = mat(
//     1, 1 / (2 eta_1), 0, 0;
//     2 eta _2 + (2 eta _2)^(n) n epsilon^"p"^(n-1), -1, 0, 0;
//     2 eta _2 + (2 eta _2)^(n) n epsilon^"p", -1, 0, 0;
//   )
// $

// or 
// $
//   tau = sum_i^N^p tau_i = 2 eta _2 epsilon^("p") + (2 eta _2 epsilon^("p"))^n
// $

// $
//   x = mat(
//     tau;
//     P;
//     epsilon^("pl");
//   )
// $
// $
//   r = mat(
//     -epsilon + 1 / (2 eta) tau + 1 / (2 G Delta t) tau + epsilon^("pl");
//     -theta + 1/(K Delta t)(P-P^"o");
//     -tau + (2 eta epsilon^("pl"))^(1/n);
//   )
// $

// $
//   J = mat(
//     1 / (2 eta) + 1 / (2 G Delta t), 0,                                         1;
//     0,                       P^"o"/(K Delta t),                                        0;
//    -1,                       0,                 ((2 eta)^(1/n) epsilon^"pl"^(1/n-1)) / n;
//   )
// $
// 
// 
// 
// 