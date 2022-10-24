module LinearPlasmaDistributionFunctions

using DualNumbers, LinearMaxwellVlasov, LinearAlgebra, StaticArrays

const LMV = LinearMaxwellVlasov

function linearelectricfield(Λ::AbstractMatrix)
  @assert size(Λ) == (3, 3)
  n = nullspace(Λ)
  if length(n) == 3 # maybe check cond(Λ) == Inf?
    return n
  else
    _, _, Vt = svd(Λ) # best approximation of nullspace for near singular matrix
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
  return @SVector [kz * v⊥ * cos(ϕ), kz * v⊥ * sin(ϕ), ω - k⊥ * v⊥]
end

function lperpendicular(op::Operator, vz::Number, v⊥::Number, ϕ::Number)
  Ω, ω, kz, k⊥ = unpack(op)
  return @SVector [(ω - kz * vz) * cos(ϕ), (ω - kz * vz) * sin(ϕ), k⊥ * vz * cos(ϕ)]
end

function (op::Operator)(s::LMV.AbstractKineticSpecies,
    electricfield::AbstractMatrix, n::Int, vz::Number, v⊥::Number, ϕ::Number)
  Ω, ω, kz, k⊥ = unpack(op)
  prefactor = - im * Ω / ω / (ω - k⊥ * v⊥ * cos(ϕ) - kz * vz - n * Ω)
  dfdvz = DualNumbers.dualpart(s(Dual(vz, 1), v⊥))
  dfdv⊥ = DualNumbers.dualpart(s(vz, Dual(v⊥, 1)))
  L = dfdvz * lparallel(op, vz, v⊥, ϕ) + dfdv⊥ * lperpendicular(op, vz, v⊥, ϕ)
  return prefactor * dot(L, electricfield)
end

function f₁xyz(op::Operator, species::LMV.AbstractKineticSpecies,
               Nvxyz=128; gyroharmonics=-10:10)
  N = Nvxyz
  vs = range(-12 * vth, 12 * vth, N)
  f1 = zeros(ComplexF64, N, N, N, length(gyroharmonics))
  for (l, n) in enumerate(gyroharmonics)
    for (k, vz) in enumerate(vs), (j, vy) in enumerate(vs), (i, vx) in enumerate(vs)
      v⊥ = sqrt(vx^2 + vy^2)
      ϕ = atan(vy, vx)
      f1[i, j, k, l] = op(species, E, n, vz, v⊥, ϕ) * exp(-im * n * ϕ)
    end
  end
  f1 ./= maximum(abs, f1)
  return vs, f1
end

function moment(vs, f1, kernel)
  output = zero(kernel(0.0, 0.0, 0.0))
  dv = vs[2] - vs[1]
  for (k, vz) in enumerate(vs), (j, vy) in enumerate(vs), (i, vx) in enumerate(vs)
    output += f1[i, j, k] * kernel(vx, vy, vz) * dv^3
  end
  return output
end

end # module LinearPlasmaDistributionFunctions
