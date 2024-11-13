##################
# INITIALISATION #
##################

using MPSKit
using KrylovKit
using Printf
using Plots
using Revise
using DrWatson

@quickactivate "Hubbard"

include(projectdir("src", "HubbardFunctions.jl"))
import .HubbardFunctions as hf


#################
# DEFINE SYSTEM #
#################

P=1;
Q=1;

t=[1.0, 0.1];
u=[8.0];
μ=0.0;

s = 2.0;
bond_dim = 20;

model = hf.OB_Sim(t, u, μ, P, Q, s, bond_dim; spin=false);
dictionary = hf.produce_groundstate(model; force=false);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
sum(real(E0))/length(H)


########################
# COMPUTE GROUNDSTATES #
########################

dictionary = hf.produce_groundstate(model; force=false);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))/length(H)

println("Groundstate energy: $E")
println("Bond dimension: $(dim_state(ψ₀))")


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

println("Exciton for s=$s: $(real(Es[1,1]))")

gap, k = hf.produce_bandgap(model)

println("Band Gap for s=$s: $gap eV at momentum $k")