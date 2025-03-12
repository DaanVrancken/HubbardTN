##################
# INITIALISATION #
##################

using DrWatson
@quickactivate "Hubbard"

using MPSKit
using KrylovKit

include(projectdir("src", "HubbardFunctions.jl"))
import .HubbardFunctions as hf


#################
# DEFINE SYSTEM #
#################

s = 2.5             # Schmidt cut value, determines bond dimension.
P = 1;              # Filling of P/Q. P/Q = 1 is half-filling.
Q = 1;
bond_dim = 20;      # Initial bond dimension of the state. Impact on result is small as DMRG modifies it.

# Define hopping, direct interaction, and chemical potential.
t=[1.0, 0.1];
u=[8.0];
μ=0.0;

# Spin=false will use SU(2) spin symmetry, the exact spin configuration cannot be deduced.
Spin = false

model = hf.OB_Sim(t, u, μ, P, Q, s, bond_dim; spin=Spin);


########################
# COMPUTE GROUNDSTATES #
########################

dictionary = hf.produce_groundstate(model; force=false);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))/length(H);

println("Groundstate energy: $E")
println("Bond dimension: $(hf.dim_state(ψ₀))")


########################
# COMPUTE EXCITATIONS #
########################

resolution = 5;
momenta = range(0, π, resolution);
nums = 1;

exc = hf.produce_excitations(model, momenta, nums; charges=[0,0.0,0]);
Es = exc["Es"];
println("Excitation energies: ")
println(Es)

println("Exciton energy for s=$s: $(real(Es[1,1]))")

gap, k = hf.produce_bandgap(model)

println("Band Gap for s=$s: $gap eV at momentum $k")