##################
# INITIALISATION #
##################

using MPSKit
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

P = 2;
Q = 3;
Bands = 3;
bond_dim = 25;

t_OS = zeros(Bands,Bands);
t_IS1 = [-0.115 0.0 0.0; 0.0 0.149 0.0; 0.0 0.0 0.159];
t = cat(t_OS, t_IS1, dims=2);

u = [1.013 0.0 0.0; 0.0 0.948 0.0; 0.0 0.0 1.211];
μ = zeros(Bands);
J = zeros(Bands,Bands);

model = hf.MB_Sim(t, u, J, μ, P, Q, 2.7, bond_dim; code = name);


########################
# COMPUTE GROUNDSTATES #
########################

dictionary = hf.produce_groundstate(model);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))./length(H);
println("Groundstate energy: $E")


########################
# COMPUTE EXCTITATIONS #
########################

resolution = 10;
momenta = range(0, π, resolution);
nums = 1;

exc = hf.produce_excitations(model, momenta, nums; charges=[0,0.0,2]);
Es = exc["Es"];
println("Excitation energies: ")
println(Es)

hf.plot_excitations(momenta,Es; title="Energy levels MIL-53(V)")

savefig(joinpath(projectdir("plots","Data_MB"),name*".pdf"))