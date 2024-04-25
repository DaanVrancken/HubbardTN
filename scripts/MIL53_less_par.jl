##################
# INITIALISATION #
##################

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
using Plots.PlotMeasures

@quickactivate "Hubbard"

include(projectdir("src", "Functions_multiband.jl"))
import .Functions_multiband as fm

# Extract name of the current file. Will be used as code name for the simulation.
name_jl = last(splitpath(Base.source_path()))
name = first(split(name_jl,"."))


#################
# DEFINE SYSTEM #
#################

P = 2;
Q = 3;
Bands = 3;
bond_dim = 30;

t_OS = zeros(Bands,Bands);
t_IS1 = [-0.115 0.0 0.0; 0.0 0.149 0.0; 0.0 0.0 0.159];
t = cat(t_OS, t_IS1, dims=2);

u = [1.013 0.0 0.0; 0.0 0.948 0.0; 0.0 0.0 1.211];
μ = zeros(Bands);
J = zeros(Bands,Bands);

model = fm.Hubbard_MB_Simulation(t, u, J, μ, P, Q, 2.7, bond_dim; code = name);


########################
# COMPUTE GROUNDSTATES #
########################

dictionary = fm.produce_groundstate(model);
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

exc = fm.produce_excitations(model, momenta, nums);
Es = exc["Es"];
println("Excitation energies: ")
println(Es)

code = get(model.kwargs, :code, "bands=2");
plot(momenta,real(Es[:,1]), label="", linecolor=:blue, title="Energy levels MIL-53(V)", left_margin = [10mm 0mm])
for i in 2:nums
    plot!(momenta,real(Es[:,i]), label="", linecolor=:blue)
end
xlabel!("k")
ylabel!("Energy density")

name = code*".pdf";
savefig(joinpath(projectdir("plots","Data_MB"),name))
