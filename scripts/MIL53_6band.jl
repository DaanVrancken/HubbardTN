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

model = fm.Hubbard_MB_Simulation(t, u, J, μ, P, Q, 2.5, bond_dim; code = name);


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

#=
resolution = 6;
momenta = range(0, π, resolution);
nums = 1;

exc_pos = fm.produce_excitations_pos(model, momenta, nums);
Es_pos = exc_pos["Es_pos"];
println("Excitation energies pos: ")
println(Es_pos)
exc_neg = fm.produce_excitations_neg(model, momenta, nums);
Es_neg = exc_neg["Es_neg"];
println("Excitation energies neg: ")
println(Es_neg)

plot(momenta./length(H),real(Es_pos[:,1]), label="", linecolor=:blue, title="Energy levels MIL-53(V)",left_margin = [10mm 0mm])
for i in 2:nums
    plot!(momenta./length(H),real(Es_pos[:,i]), label="", linecolor=:blue)
end
for i in 1:nums
    plot!(momenta./length(H),real(Es_neg[:,i]), label="", linecolor=:red)
end
xlabel!("k")
ylabel!("Energy density")

code = get(model.kwargs, :code, "MIL53");
savefig(projectdir()*"//plots//"*code*".pdf")
=#