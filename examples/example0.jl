using GLMakie

using LinearMaxwellVlasov, LinearAlgebra, LinearPlasmaDistributionFunctions
const LPDF = LinearPlasmaDistributionFunctions

function foo(ngyro=30)
  M = rand(3, 3); M .-= M' # a plasma tensor, with a nullspace
  E = LPDF.linearelectricfield(M)
  Ω = 1.0
  Π = 1.0
  ω = 10 * rand() * (1 + im / 10)
  kz = 2rand() - 1
  k⊥ = rand()
  gyroharmonics = range(-ngyro, ngyro)
  op = LPDF.Operator(Ω, ω, kz, k⊥, E, gyroharmonics)
  vth = rand()
  species = MaxwellianSpecies(Π, Ω, vth)

  vs, f0, f1 = LPDF.f₁vxvyvz(op, species, N=128, vmaxmultiplier=0.5)
  @show -im * ω * LPDF.zerothmoment(op, species)
  @show -im * ω * LPDF.firstmoment(op, species)
  @show -im * ω * LPDF.secondmoment(op, species)

  return vs, f0, f1
end

vs, f0, f1 = foo(10)

f0 ./= maximum(abs, f0)
f1 ./= maximum(abs, f1)

f11 = real.(sum(f1, dims=4))[:, :, :, 1]

contour(vs, vs, vs, f11,  isovalue=0.01, alpha=0.2)

#contour(vs, vs, vs, f0 .+ f11 / 100,  isovalue=0.01, alpha=0.2)

