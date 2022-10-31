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

struct Operator{TΩ, Tω, TKz, TK⊥, TE}
  Ω::TΩ
  ω::Tω
  kz::TKz
  k⊥::TK⊥
  E::TE
end
function Operator(species::LMV.AbstractKineticSpecies,
                  config::LMV.Configuration,
                  EorΛ::AbstractArray)
  E = if size(EorΛ) == (3, 3)
    linearelectricfield(EorΛ)
  elseif size(EorΛ) == (3,)
    EorΛ
  end
  return Operator(species.Ω,
                  config.frequency,
                  LMV.parallel(options.waveumber),
                  LMV.perpendicular(options.waveumber),
                  E)
end
unpack(op::Operator) = (op.Ω, op.ω, op.kz, op.k⊥, op.E)

function lparallel(op::Operator, vz::Number, v⊥::Number, ϕ::Number)
  Ω, ω, kz, k⊥, _ = unpack(op)
  return @SVector [kz * v⊥ * cos(ϕ), kz * v⊥ * sin(ϕ), ω - k⊥ * v⊥]
end

function lperpendicular(op::Operator, vz::Number, v⊥::Number, ϕ::Number)
  Ω, ω, kz, k⊥, _ = unpack(op)
  return @SVector [(ω - kz * vz) * cos(ϕ), (ω - kz * vz) * sin(ϕ), k⊥ * vz * cos(ϕ)]
end

function (op::Operator)(s::LMV.AbstractKineticSpecies,
    vz::Number, v⊥::Number, ϕ::Number, gyroharmonics)
  n = gyroharmonics # reference by short hand variable name
  Ω, ω, kz, k⊥, E = unpack(op)
  prefactor = @. - im * Ω / ω / (ω - k⊥ * v⊥ * cos(ϕ) - kz * vz - n * Ω)
  dfdvz = DualNumbers.dualpart(s(Dual(vz, 1), v⊥))
  dfdv⊥ = DualNumbers.dualpart(s(vz, Dual(v⊥, 1)))
  L = dfdvz * lparallel(op, vz, v⊥, ϕ) + dfdv⊥ * lperpendicular(op, vz, v⊥, ϕ)
  dotLE = dot(L, E)
  return @. prefactor * dotLE * cis(n * ϕ)
end

function (op::Operator)(s::LMV.AbstractKineticSpecies,
    vz::Number, v⊥::Number, gyroharmonics; atol=0, rtol=sqrt(eps()))
  return HCubature.hcubature(ϕ->op(s, vz, v⊥, ϕ[1], gyroharmonics), (-π,), (π,), 
                       atol=atol, rtol=rtol)[1]
end

function normalise(x)
  maxabsx = maximum(abs, x)
  iszero(maxabsx) && return x
  return x ./ maxabsx
end

function f₀(species, vzs, v⊥s)
  f0 = zeros(Float64, length(vzs), length(v⊥s))
  for (j, v⊥) in enumerate(v⊥s), (i, vz) in enumerate(vzs)
    f0[i, j] = species(vz, v⊥)
  end
  return f0
end

function f₁vzv⊥n(op::Operator, species::LMV.AbstractKineticSpecies,
    gyroharmonics; N=64)
  vz0, v⊥0 = LMV.lowerintegralbounds(species)
  vz1, v⊥1 = LMV.upperintegralbounds(species)
  vzs = range(vz0, vz1, N)
  v⊥s = range(v⊥0, v⊥1, N)
  ns = gyroharmonics # short hand reference
  f1 = zeros(ComplexF64, N, N, length(ns))
  for (j, v⊥) in enumerate(v⊥s), (i, vz) in enumerate(vzs)
    # iteration is in wrong order but loop is hot enough to be able ignore?
    f1[i, j, :] .= op(species, vz, v⊥, ns)
  end
  f0 = f₀(species, vzs, v⊥s)
  return vzs, v⊥s, ns, normalise(f0), normalise(f1)
end

function f₁vzv⊥ϕ(op::Operator, species::LMV.AbstractKineticSpecies,
    gyroharmonics; N=64)
  vzs, v⊥s, ns, f0, f1n = f₁vzv⊥n(op, species, gyroharmonics; N=64)
  ϕs = range(-π, π, N)
  f1 = zeros(ComplexF64, N, N, N)
  for (k, ϕ) in enumerate(ϕs), (j, v⊥) in enumerate(v⊥s), (i, vz) in enumerate(vzs)
    f1[i, j, k] += dot((@view f1n[i, j, :]), cis.(-gyroharmonics .* ϕ))
  end
  return vzs, v⊥s, ϕs, f0, normalise(f1)
end

function moment(op::Operator, s::LMV.AbstractKineticSpecies,
    kernel::AbstractMomentOperator, gyroharmonics; atol=0, rtol=sqrt(eps()))
  function integrand(vzv⊥ϕ)
    vz, v⊥, ϕ = vzv⊥ϕ
    return v⊥ * kernel(vzv⊥ϕ) * sum(op(s, vz, v⊥, ϕ, gyroharmonics))
  end
  vz⊥lo = LMV.lowerintegralbounds(s)
  vz⊥hi = LMV.upperintegralbounds(s)
  return 2π * HCubature.hcubature(integrand, (vz⊥lo..., -π), (vz⊥hi..., π),
                                  atol=atol, rtol=rtol)[1]
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
  @eval function $(Symbol(string(stub) * "moment"))(op::Operator,
      s::LMV.AbstractKineticSpecies, gyroharmonics; atol=0, rtol=sqrt(eps()))
    return moment(op, s, $(amo)(), gyroharmonics; atol=atol, rtol=rtol)
  end
end

function integrateover(op::Operator, s::LMV.AbstractKineticSpecies,
    directions::Vararg{Symbol, N}; atol=0, rtol=sqrt(eps())) where N
  vz⊥lo = LMV.lowerintegralbounds(s)
  vz⊥hi = LMV.upperintegralbounds(s)
  function integrand(vzv⊥ϕ)
    vz, v⊥, ϕ = vzv⊥ϕ
    return v⊥ * sum(op(s, vz, v⊥, ϕ, gyroharmonics))
  end
  @assert mapreduce(i->i ∈ (:vz, :v⊥, :ϕ), &, directions)
  if N == 1
    if :vz ∈ directions
      return v⊥ϕ -> 2π * HCubature.hcubature(vz->integrand((vz[1], v⊥ϕ...)),
        (vz⊥lo[1],), (vz⊥hi[1],), atol=atol, rtol=rtol)[1]
    elseif :v⊥ ∈ directions
      return vzϕ -> 2π * HCubature.hcubature(v⊥->integrand((vzϕ[1], v⊥[1], vzϕ[2])),
        (vz⊥lo[2],), (vz⊥hi[2],), atol=atol, rtol=rtol)[1]
    elseif :ϕ ∈ directions
      return vzv⊥ -> 2π * HCubature.hcubature(ϕ->integrand((vzv⊥..., ϕ[1])),
        (-π,), (π,), atol=atol, rtol=rtol)[1]
    end
  elseif N == 2
    if :vz ∈ directions && :v⊥ ∈ directions
      return ϕ -> 2π * HCubature.hcubature(vz⊥->integrand((vz⊥[1], vz⊥[2], ϕ)),
        vz⊥lo, vz⊥hi, atol=atol, rtol=rtol)[1]
    elseif :vz ∈ directions && :ϕ ∈ directions
      return v⊥ -> 2π * HCubature.hcubature(vzϕ->integrand((vzϕ[1], v⊥, vzϕ[2])),
        (vz⊥lo[1], -π), (vz⊥hi[1], π), atol=atol, rtol=rtol)[1]
    elseif :vz ∈ directions && :ϕ ∈ directions
      return vz -> 2π * HCubature.hcubature(v⊥ϕ->integrand(v⊥ϕ->(vz, v⊥ϕ...)),
        (vz⊥lo[2], -π), (vz⊥hi[2], π), atol=atol, rtol=rtol)[1]
    end
  else
    @error "Directions $directions must be 1 or 2 of (:vz, :v⊥, :ϕ)."
  end
end

end # module LinearPlasmaDistributionFunctions
