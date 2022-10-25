module LinearPlasmaDistributionFunctions

using DualNumbers, LinearMaxwellVlasov, LinearAlgebra, StaticArrays
using HCubature

const LMV = LinearMaxwellVlasov

abstract type AbstractMomentOperator end

(amo::AbstractMomentOperator)(vzv⊥ϕ) = amo(vzv⊥ϕ...)

function linearelectricfield(Λ::AbstractMatrix)
  @assert size(Λ) == (3, 3)
  n = nullspace(Λ)
  if length(n) == 3 # maybe check cond(Λ) == Inf?
    return n
  else
    _, σ, Vt = svd(Λ) # best approximation of nullspace for near singular matrix
    @warn "No nullspace found. Instead using the vector in the complex unitary
    matrix associated with smallest singular value, $(σ[3]),
    from SVD decomposition."
    return Vt[:, 3]
  end
end

struct Operator{TΩ, Tω, TKz, TK⊥, TE, TG}
  Ω::TΩ
  ω::Tω
  kz::TKz
  k⊥::TK⊥
  E::TE
  gyroharmonics::TG
end
function Operator(species::LMV.AbstractKineticSpecies,
                  config::LMV.Configuration,
                  solutiontensor::AbstractMatrix,
                  gyroharmonics)
  return Operator(species.Ω,
                  config.frequency,
                  LMV.parallel(options.waveumber),
                  LMV.perpendicular(options.waveumber),
                  linearelectricfield(solutiontensor),
                  gyroharmonics)
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
    n::Int, vz::Number, v⊥::Number, ϕ::Number)
  Ω, ω, kz, k⊥ = unpack(op)
  prefactor = - im * Ω / ω / (ω - k⊥ * v⊥ * cos(ϕ) - kz * vz - n * Ω)
  dfdvz = DualNumbers.dualpart(s(Dual(vz, 1), v⊥))
  dfdv⊥ = DualNumbers.dualpart(s(vz, Dual(v⊥, 1)))
  L = dfdvz * lparallel(op, vz, v⊥, ϕ) + dfdv⊥ * lperpendicular(op, vz, v⊥, ϕ)
  electricfield = op.E
  return prefactor * dot(L, electricfield)
end

function f₁vzv⊥ϕ(op::Operator, species::LMV.AbstractKineticSpecies;
    N=64, ϕ=0)
  vz0, v⊥0 = LMV.lowerintegralbounds(species)
  vz1, v⊥1 = LMV.upperintegralbounds(species)
  vzs = range(vz0, vz1, N)
  v⊥s = range(v⊥0, v⊥1, N)
  f1 = zeros(ComplexF64, N, N, length(op.gyroharmonics))
  for (k, n) in enumerate(op.gyroharmonics)
    for (j, v⊥) in enumerate(v⊥s), (i, vz) in enumerate(vzs)
      f1[i, j, k] = op(species, n, vz, v⊥, ϕ) * cis(n * ϕ)
    end
  end
  f0 = zeros(Float64, N, N)
    for (k, vz) in enumerate(vzs), (j, v⊥) in enumerate(v⊥s)
    f0[i, j] = species(vz, v⊥)
  end
  f1 ./= maximum(abs, f1)
  return vs, f0, f1
end

function f₁vxvyvz(op::Operator, species::LMV.AbstractKineticSpecies;
    N=128, vmaxmultiplier=1)
  vmax1 = maximum(abs, LMV.lowerintegralbounds(species))
  vmax2 = maximum(abs, LMV.upperintegralbounds(species))
  vmax = max(vmax1, vmax2) * vmaxmultiplier
  vs = range(-vmax, vmax, N)
  f1 = zeros(ComplexF64, N, N, N, length(op.gyroharmonics))
  for (l, n) in enumerate(op.gyroharmonics)
    for (k, vz) in enumerate(vs), (j, vy) in enumerate(vs), (i, vx) in enumerate(vs)
      v⊥ = sqrt(vx^2 + vy^2)
      ϕ = atan(vy, vx)
      f1[i, j, k, l] = op(species, n, vz, v⊥, ϕ) * cis(n * ϕ)
    end
  end
  f0 = zeros(Float64, N, N, N)
  for (k, vz) in enumerate(vs), (j, vy) in enumerate(vs), (i, vx) in enumerate(vs)
    v⊥ = sqrt(vx^2 + vy^2)
    ϕ = atan(vy, vx)
    f0[i, j, k] = species(vz, v⊥)
  end
  f1 ./= maximum(abs, f1)
  return vs, f0, f1
end

function moment(op::Operator, s::LMV.AbstractKineticSpecies,
    kernel::AbstractMomentOperator; atol=0, rtol=sqrt(eps()))
  function integrand(vzv⊥ϕ)
    vz, v⊥, ϕ = vzv⊥ϕ
    return v⊥ * mapreduce(n->kernel(vzv⊥ϕ) * op(s, n, vz, v⊥, ϕ) * cis(n * ϕ),
                          +, op.gyroharmonics)
  end
  vz⊥lo = LMV.lowerintegralbounds(s)
  vz⊥hi = LMV.upperintegralbounds(s)
  return 2π * HCubature.hcubature(integrand, (vz⊥lo..., -π), (vz⊥hi..., π))[1]
end

struct ZerothMomentOperator <: AbstractMomentOperator end
(op::ZerothMomentOperator)(vzv⊥ϕ) = 1

struct FirstMomentOperator <: AbstractMomentOperator end
function (op::FirstMomentOperator)(vzv⊥ϕ)
  vz, v⊥, ϕ = vzv⊥ϕ
  return @SVector [v⊥ * cos(ϕ), v⊥ * sin(ϕ), vz]
end

struct SecondMomentOperator <: AbstractMomentOperator end
(op::SecondMomentOperator)(vzv⊥ϕ) = (vzv⊥ϕ[1]^2 + vzv⊥ϕ[2]^2)

for (stub, amo) ∈ ((:zeroth, :ZerothMomentOperator),
                   (:first, :FirstMomentOperator),
                   (:second, :SecondMomentOperator))
  @eval function $(Symbol(string(stub) * "moment"))(op, s; atol=0, rtol=sqrt(eps()))
    return moment(op, s, $(amo)(); atol=atol, rtol=rtol)
  end
end

end # module LinearPlasmaDistributionFunctions
