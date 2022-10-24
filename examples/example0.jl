using GLMakie

using LinearMaxwellVlasov, LinearAlgebra, LinearPlasmaDistributionFunctions

function foo()
M = rand(3, 3); M .-= M' # a plasma tensor

E = LinearPlasmaDistributionFunctions.linearelectricfield(M)
Ω = 1.0
Π = 1.0
ω = 10 * rand() * (1 + im / 10)
kz = 2rand() - 1
k⊥ = rand()
op = LinearPlasmaDistributionFunctions.Operator(Ω, ω, kz, k⊥)
vth = rand()
species = MaxwellianSpecies(Π, Ω, vth)


N = 128
vs = range(-6 * vth, 6 * vth, N)
gyroharmonics = range(-30, 30)
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

vs, f1 = foo()
f11 = real.(sum(f1, dims=4))[:, :, :, 1]
contour(vs, vs, vs, f11,  isovalue=0.01, alpha=0.2)
