##################
# INITIALISATION #
##################

using MPSKit
using Printf
using Revise
using Plots
using DrWatson

@quickactivate "Hubbard"

include(projectdir("src", "HubbardFunctions.jl"))
import .HubbardFunctions as hf


#################
# DEFINE SYSTEM #
#################

μ = 1.0;
P=1; 
Q=1;

model = hf.OBC_Sim([1.0], [1.0], P, Q, 2.5; verbosity_mu=1);


###############
# GROUNDSTATE #
###############

dictionary = hf.produce_groundstate(model; force=true);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];

Ne = hf.density_state(ψ₀);
E0 = sum(expectation_value(ψ₀, H)) + dictionary["μ"]*Ne;

println("Electron density: $Ne")
println("Energy density: $E0")


########################
# COMPUTE EXCTITATIONS #
########################

μ= 3.535;
u=[7.658];
t=[2.726];

model = hf.OBC_Sim(t, u, μ, 2.0; verbosity=0, tol_mu=1e-8, maxiter_mu=30);

dictionary = hf.produce_groundstate(model; force=true);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
Ne = hf.density_state(ψ₀);
E0 = sum(expectation_value(ψ₀, H)) + dictionary["μ"]*Ne

println("Electron density: $Ne")
println("Energy density: $E0")

nums = 1;
resolution = 5;
momenta = range(0, π, resolution);

exc = hf.produce_excitations(model, momenta, nums; force=true);
Es = exc["Es"];


#########
# Tools #
#########

D = hf.dim_state(ψ₀)
println("Bond dimensions: ")
println(D)


############
# PLOTTING #
############

hf.plot_excitations(momenta,Es)