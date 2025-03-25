using Chairmarks, InteractiveUtils

# # Problem to solve
# 
# $f = \begin{pmatrix}
# foo(\sum_i r_i) \\
# bar(\sum_i r_i) \\
# \end{pmatrix}$

# # Define rheologies, functions & some data

abstract type AbstractRheology end

struct Rheology1 <: AbstractRheology
    a::Float64
    b::Float64
end

struct Rheology2 <: AbstractRheology
    a::Float64
    b::Float64
end

foo(r::AbstractRheology, x) = r.a * x + r.b
bar(r::AbstractRheology, x) = r.a * x^3 / r.b

x   = rand() 
r   = Rheology1(1.0, 2.0), Rheology2(3.0, 4.0)
fns = foo, bar

# # Naive implementation

function f1(r, x, fns) 
    sol = Array{Float64}(undef, length(fns))
    for (i, fn) in enumerate(fns), rᵢ in r
        sol[i] += fn(rᵢ , x)
    end 
    sol
end

f1(r, x, fns)
#
@b f1($(r, x, fns)...)
#
@code_warntype f1(r, x, fns)
#

function f2(r::NTuple{N1,Any}, x::T, fns::NTuple{N2,Any}) where {N1,N2,T}
    sol = ntuple(N2) do i 
        sum(fns[i](rᵢ , x) for rᵢ in r)
    end
end

f2(r, x, fns)
#
@b f2($(r, x, fns)...)
#
@code_warntype f2(r, x, fns)
#
foobar(r::AbstractRheology, x)  = foo(r, x) + bar(r, x)
barfoo(r::AbstractRheology, x)  = foo(r, x)^bar(r, x)
#
r = Rheology1(1.0, 2.0), Rheology2(3.0, 4.0), Rheology1(1.0, 2.0), Rheology2(3.0, 4.0), Rheology2(3.0, 4.0), Rheology1(1.0, 2.0), Rheology2(3.0, 4.0), Rheology2(3.0, 4.0), Rheology1(1.0, 2.0), Rheology2(3.0, 4.0)
fns = foo, foobar, foobar, bar, foo, bar, barfoo, bar, foo, bar, barfoo
#
@b f2($(r, x, fns)...)
#
@code_warntype f2(r, x, fns)
#
function f3(r::NTuple{N1,Any}, x::T, fns::NTuple{N2,Any}) where {N1,N2,T}
    sol = ntuple(N2) do i 
        @inline 
        tmp = zero(T)
        ntuple(N1) do j
            @inline 
            tmp += fns[i](r[j] , x)
        end
    end
end

@b f3($(r, x, fns)...)
#
@code_warntype f3(r, x, fns)
#
function f3(r::NTuple{N1,Any}, x::T, fns::NTuple{N2,Any}) where {N1,N2,T}
    sol = ntuple(N2) do i 
        @inline 
        tmp = Ref(zero(T))
        ntuple(N1) do j
            @inline 
            tmp[] += fns[i](r[j] , x)
        end
    end
end

@b f3($(r, x, fns)...)
#
@code_warntype f3(r, x, fns)
#
@generated function f4(r::NTuple{N1,Any}, x::T, fns::NTuple{N2,Any}) where {N1,N2,T}
    quote
        Base.Cartesian.@ntuple $N2 i -> begin
            tmp = Base.Cartesian.@ntuple $N1 j -> begin
                fns[i](r[j] , x)
            end
            sum(tmp)
        end
    end
end
 
@b f4($(r, x, fns)...)
#
@code_warntype f4(r, x, fns)
#