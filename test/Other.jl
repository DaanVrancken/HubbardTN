using ThreadPinning
using LinearAlgebra
using MPSKit, MPSKitModels
using TensorKit
using KrylovKit
using DataFrames
using DrWatson
using Plots, StatsPlots
using Plots.PlotMeasures
using TensorOperations
using Revise

include("HubbardFunctions.jl")
import .HubbardFunctions as hf

model = hf.OB_Sim([1.0],[8.0], 0.0,1,1,2.0;spin=true)

dictionary = hf.produce_groundstate(model;force=false);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))./length(H);
println("Groundstate energy: $E")

hf.density_state(model)
hf.density_spin(model)

resolution = 5;
momenta = range(0, π, resolution);
nums = 1;

exc = hf.produce_excitations(model, momenta, nums; charges=[0,0.0,1]);
Es = exc["Es"];
println("Excitation energies: ")
println(Es)

hf.plot_excitations(momenta,Es; title="Energy levels MIL-53(V)")