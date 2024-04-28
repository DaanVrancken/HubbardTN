##################
# INITIALISATION #
##################

using MPSKit
using Printf
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

t_OS = [0.0 0.0; 0.0 0.0];
t_IS = [1.0 0.0; 0.0 1.0];
t = cat(t_OS,t_IS, dims=2)

U = [1.0 0.0; 0.0 1.0]
V = [0.0 0.0; 0.0 0.0]
u = cat(U,V,dims=2)

J = [0.0 0.0; 0.0 0.0]

μ = [0.5, 0.5]

bond_dim = 20;

model = hf.MBC_Sim(t, u, J, μ, 2.0, bond_dim; verbosity=0, code=name);


#######################
# COMPUTE GROUNDSTATE #
#######################

dictionary = hf.produce_groundstate(model; force=true);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
Ne = hf.density_state(ψ₀);
E0 = expectation_value(ψ₀, H) + μ.*Ne;
E = sum(real(E0))/length(H)


########################
# COMPUTE EXCTITATIONS #
########################

resolution = 5;
momenta = range(0, π, resolution);
nums = 1;

exc = hf.produce_excitations(model, momenta, nums; force=true);
Es = exc["Es"];
println("Excitation energies: ")
println(Es)


#########
# Tools #
#########

D = hf.dim_state(ψ₀)
println("Bond dimensions: ")
println(D)


############
# PLOTTING #
############

hf.plot_excitations(momenta,Es; title="Energy levels MIL-53(V)")