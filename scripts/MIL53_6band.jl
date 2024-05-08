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

P = 2;
Q = 3;
Bands = 6;
bond_dim = 25;

t_OSa = [0.0 0.067 0.024; 0.067 0.0 -0.016; 0.024 -0.016 0.0];
t_OSb = [0.0 -0.071 0.015; -0.071 0.0 0.042; 0.015 0.042 0.0];
t_IS1a = [-0.115 -0.03 -0.042; 0.036 0.149 0.088; -0.038 -0.0977 0.159];
Z = zeros(Bands÷2,Bands÷2);
t_OS = [t_OSa t_IS1a; t_IS1a' t_OSb];
t_IS = [Z Z; t_IS1a' Z];
t = cat(t_OS, t_IS, dims=2);

μ = zeros(Bands);

U_OS = [1.013 0.303 0.363; 0.303 0.948 0.323; 0.363 0.323 1.211];
U_IS = [0.476 0.309 0.361; 0.302 0.284 0.284; 0.368 0.292 0.350];
u_OS = [U_OS U_IS; U_IS' U_OS];
u_IS = [Z Z; U_IS' Z];
u = cat(u_OS, u_IS, dims=2);

J_OS = [0.0 0.337 0.316; 0.337 0.0 0.340; 0.316 0.340 0.0];
J = [J_OS Z; Z J_OS];

model = hf.MB_Sim(t, u, J, μ, P, Q, 1.7, bond_dim; code = name);


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

resolution = 5;
momenta = range(0, π, resolution);
nums = 1;

momentum = 0.0

exc = hf.produce_excitations(model, momentum, nums);
Es = exc["Es"];
println("Excitation energies: ")
println(Es)

hf.plot_excitations(momenta,Es; title="Energy levels MIL-53(V)")

savefig(joinpath(projectdir("plots","Data_MB"),name*".pdf"))