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

@quickactivate "Hubbard"

include(projectdir("Src", "Functions_multiband.jl"))
import .Functions_multiband as fm

name_jl = split(@__FILE__, "/")
name = first(split(last(name_jl),"."))

P = 2;
Q = 3;
Bands = 3;
# max bond dim in initialisation is 20 (taking it larger does not improve it)
bond_dim = 30;

t_OS = zeros(Bands,Bands);
t_IS1 = [-0.115 0.0 0.0; 0.0 0.149 0.0; 0.0 0.0 0.159];
t = cat(t_OS, t_IS1, dims=2);

u = [1.013 0.0 0.0; 0.0 0.948 0.0; 0.0 0.0 1.211];
μ = zeros(Bands);
J = zeros(Bands,Bands);

model = fm.Hubbard_MB_Simulation(t, u, J, μ, P, Q, 2.7, bond_dim; verbosity=0, code = name);

dictionary = fm.produce_groundstate(model);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))./length(H);
println("Groundstate energy: $E")

resolution = 12;
momenta = range(0, π, resolution);
nums = 7;

exc_pos = fm.produce_excitations_pos(model, momenta, nums);
Es_pos = exc_pos["Es_pos"];
println("Excitation energies pos: ")
println(Es_pos)
#=
exc_neg = fm.produce_excitations_neg(model, momenta, nums);
Es_neg = exc_neg["Es_neg"];
println("Excitation energies neg: ")
println(Es_neg)
=#

plot(momenta./length(H),real(Es_pos[:,1]), label="", linecolor=:blue, title="Energy levels MIL-53(V)",left_margin = [15mm 0mm], bottom_margin = [10mm 0mm])
for i in 2:nums
    plot!(momenta./length(H),real(Es_pos[:,i]), label="", linecolor=:blue)
end
#=
for i in 1:nums
    plot!(momenta./length(H),real(Es_neg[:,i]), label="", linecolor=:red)
end
=#
xlabel!("k")
ylabel!("Energy density")

code = get(model.kwargs, :code, "MIL53");
savefig(projectdir()*"//plots//"*code*".pdf")
