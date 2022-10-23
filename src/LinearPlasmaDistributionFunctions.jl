module LinearPlasmaDistributionFunctions

using LinearMaxwellVlasov, LinearAlgebra


function linearelectricfield(M::AbstractMatrix)
  @assert size(M) == (3, 3)
  n = nullspace(M)
  if length(n) == 3 # maybe check cond(M) == Inf?
    return n
  else
    _, _, Vt = svd(M) # best approximation of nullspace for near singular matrix
    return Vt[:, 3]
  end
end

struct Operator{TΩ, Tω, TKz, TK⊥}
  Ω::TΩ
  ω::Tω
  kz::TKz
  k⊥::TK⊥
end
unpack(op::Operator) = (op.Ω, op.ω, op.kz, op.k⊥)

function lparallel(op::Operator, vz::Number, v⊥::Number, ϕ::Number)
  Ω, ω, kz, k⊥ = unpack(op)
  return @SVector [ω + kz * (v⊥ * cos(ϕ) -vz), sin(ϕ) * kz * v⊥, kz * vz]
end

function lperpendicular(op::Operator, vz::Number, v⊥::Number, ϕ::Number)
  Ω, ω, kz, k⊥ = unpack(op)
  return @SVector [(ω + kz * vz) * cos(ϕ), (ω - kz * vz) * sin(ϕ), k⊥ * vz * cos(ϕ)]
end

function (op::Operator)(s::AbstractKineticSpecies,
    electricfield::AbstractMatrix, n::Int, vz::Number, v⊥::Number, ϕ::Number)
  Ω, ω, kz, k⊥ = unpack(op)
  prefactor = - im * Ω / (ω + k⊥ * v⊥ * cos(ϕ) - kz * vz - nΩ)
  dfdvz = DualNumbers.dualpart(s(Dual(vz, 1), v⊥))
  dfdv⊥ = DualNumbers.dualpart(s(vz, Dual(v⊥, 1)))
  L = dfdvz * lparallel(op, vz, v⊥, ϕ) + dfdv⊥ * hperpendicular(op, vz, v⊥, ϕ)
  return prefactor * dot(L, E)
end

end # module LinearPlasmaDistributionFunctions
