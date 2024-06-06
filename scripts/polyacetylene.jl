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

# Extract name of the current file. Will be used as code name for the simulation.
name_jl = last(splitpath(Base.source_path()))
name = first(split(name_jl,"."))


#################
# DEFINE SYSTEM #
#################

s = 2.0

t_OS = [0.0  2.588; 2.588 0.0];
t_IS = [0.0345 2.745; -0.036 0.0345];
t = cat(t_OS,t_IS, dims=2)
U = [9.86 5.7656; 5.7656 9.86];
J = [0.0 0.096; 0.096 0.0]
μ = [-0.784,-0.784]

P = 1;
Q = 1;
bond_dim = 20;

model = hf.MB_Sim(t, U, J, μ, P, Q, s, bond_dim; code = name);


########################
# COMPUTE GROUNDSTATES #
########################

dictionary = hf.produce_groundstate(model);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))./length(H);
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

println("Band Gap for s=$s: $(real(Es[1,1]))")