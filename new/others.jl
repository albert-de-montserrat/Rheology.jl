# complete flatten a tuple
@inline superflatten(t::NTuple{N, Any}) where N = superflatten(first(t))..., superflatten(Base.tail(t))... 
@inline superflatten(::Tuple{})                 = ()
@inline superflatten(x)                         = (x,)

# @generated function harmonic_average(r::NTuple{N, AbstractRheology}, fn::F, args) where {N, F}
#     quote
#         v = 0e0
#         Base.@ntuple $N i -> v += begin
#             x = inv( fn(r[i], args) )
#             x = isinf(x) ? 0e0 : x
#         end
#         return inv(v)
#     end
# end

# harmonic_average_stress(r, args) = harmonic_average(r, compute_stress, args)
# harmonic_average_strain_rate(r, args) = harmonic_average(r, compute_strain_rate, args)
