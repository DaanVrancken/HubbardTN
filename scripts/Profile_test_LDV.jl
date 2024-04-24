using LinearAlgebra
using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using KrylovKit
using OptimKit
using Printf
using Plots
using Revise
using DrWatson
using JSON
using Plots.PlotMeasures
using ThreadPinning
LinearAlgebra.BLAS.set_num_threads(1)
ThreadPinning.pinthreads(:cores)
ThreadPinning.threadinfo(; blas=true, hints=true)

@quickactivate "Hubbard"



include(projectdir("src", "Functions_multiband.jl"))
import .Functions_multiband as fm

t0 = 10.0/4.0;
t1 = 2.5/4.0;

t_OS = [0.0 t0+t1; t0+t1 0.0];
t_IS = [0.0 0.0; t0-t1 0.0];
t = cat(t_OS,t_IS, dims=2)

U = [3.0 0.0; 0.0 3.0]
V = [0.0 0.0; 0.0 0.0]
u = cat(U,V,dims=2)

J = [0.0 0.0; 0.0 0.0]

μ = [0.0, 0.0]

P = 1;
Q = 1;
if iseven(P)
    T = Q;
else 
    T = 2*Q;
end

svalue = 2.0
bond_dim = 20


model = fm.Hubbard_MB_Simulation(t, u, J, μ, P, Q, svalue, bond_dim; verbosity=0, code = "SSH_PV");
dictionary = fm.produce_groundstate(model);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))/length(H)

# inspect some of the hamiltonian properties

i = 3
Hi = H[i];
nlevels = size(Hi, 1)

length(collect(keys(Hi)))
Hi.imspaces
dim.(Hi.imspaces)

for i in eachindex(H)
    println("Hamiltonian at site $i")
    Hi = H[i]

    println("nonzeros: $(length(collect(keys(Hi))))")
    nlevels = Hi.odim 
    vspaces = Hi.imspaces
    vdims = dim.(vspaces)
    
    println("vdim: $vdims")
    # println("vspace: $vspaces")
end

# inspect some of the groundstate properties
ψ₀
for i in 1:length(ψ₀)
    V = left_virtualspace(ψ₀, i)
    Vdim = dim(V)
    println("Virtual space at site $i: $Vdim")
end

entanglementplot(ψ₀; sectorformatter = string)

i = 2
_sectors = sectors(fuse(left_virtualspace(ψ₀, i), dual(left_virtualspace(ψ₀, i)))) 
transferplot(ψ₀, sectors=_sectors; legend=:top)


# excitations

resolution = 8;
momenta = range(0, π, resolution);
nums = 8;
momentum = 0.0

Q = model.Q
sector = fℤ₂(1) ⊠ SU2Irrep(1 // 2) ⊠ U1Irrep(Q)

sector.sectors
map(x -> sprint(show, x; context= :typeinfo=>typeof(x)), sector.sectors)


# dictionary = produce_groundstate(simul)
ψ = dictionary["groundstate"]
H = dictionary["ham"]
# if trunc_dim == 0
#     envs = dictionary["environments"]
# else
#     ψ, envs = changebonds(ψ, H, VUMPSSvdCut(; trscheme=truncdim(trunc_dim)))
# end
verbosity = 5
tol_solver = 1e-5
maxiter = 50
krylovdim = 30

ϕ₀ = MPSKit.LeftGaugedQP(rand, ψ, ψ; sector, momentum)

solver = GMRES(; verbosity=0, tol=tol_solver, maxiter, krylovdim)
# solver = BiCGStab(; verbosity, tol=tol_solver, maxiter, krylovdim)

momentum1 = 0.1

eigsolver = Arnoldi(; krylovdim=50, tol=1e-5, verbosity=5, eager=true)
# eigsolver = Lanczos(; krylovdim=50, tol=1e-5, verbosity=5, eager=true)


@time E, B = excitations(H, QuasiparticleAnsatz(eigsolver), deepcopy(ϕ₀), envs, envs; num=nums, solver=solver);


ϕ₁ = MPSKit.LeftGaugedQP(ψ, ψ, B[1].VLs, B[1].Xs, momentum1);
@time E2, B2 = excitations(H, QuasiparticleAnsatz(eigsolver), deepcopy(ϕ₁), envs, envs; num=nums, solver=solver);
ϕ₁ = MPSKit.LeftGaugedQP(ψ, ψ, B2[1].VLs, B2[1].Xs, 0.15);
@time E2, B2 = excitations(H, QuasiparticleAnsatz(eigsolver), deepcopy(ϕ₁), envs, envs; num=nums, solver=solver);



envs = dictionary["environments"];

nums = 1
@time excitations(H, QuasiparticleAnsatz(eigsolver), deepcopy(ϕ₀), envs, envs; num=nums, solver=solver);

eigsolver = Arnoldi(; krylovdim=50, tol=1e-6, verbosity=5, eager=true)
@time excitations(H, QuasiparticleAnsatz(eigsolver), deepcopy(ϕ₀), envs, envs; num=nums, solver=solver);


# profiler
qp_envs(ϕ) = environments(ϕ, H, envs, envs; solver)
E = MPSKit.effective_excitation_renormalization_energy(H, ϕ₀, envs, envs)

function toprofile(ϕ)
    return MPSKit.effective_excitation_hamiltonian(H, ϕ, qp_envs(ϕ), E)
end
toprofile(ϕ₀)

@profview toprofile(ϕ₀)

@profview excitations(H, QuasiparticleAnsatz(eigsolver), deepcopy(ϕ₀), envs, envs; num=nums, solver=solver);


@profview fm.produce_excitations_pos(model, momenta, nums)
