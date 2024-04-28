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

U = [3.0 0.0; 0.0 3.0]
V = [0.0 0.0; 0.0 0.0]
u = cat(U,V,dims=2)

J = [0.0 0.0; 0.0 0.0]

μ = [0.0, 0.0]

P = 1;
Q = 1;
bond_dim = 20;

model = hf.MB_Sim(t, u, J, μ, P, Q, 2.0, bond_dim; code = name);


#######################
# COMPUTE GROUNDSTATE #
#######################

dictionary = hf.produce_groundstate(model; force=true);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))/length(H)
println("Groundstate energy: $E")

# E≈-0.6631711345220876 with quit some variation


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

# Es≈[0.4029843194152638; 0.3008443321674866; -0.07803741107392667; -0.7532301753220513; -1.1753923849821284]


#########
# Tools #
#########

D = hf.dim_state(ψ₀)
println("Bond dimensions: ")
println(D)

electron_number = hf.density_state(model)
println("Electron occupancies: ")
println(electron_number)

if sum(electron_number)/length(ψ₀)!=1
    @warn "Filling is not equal to 1!"
end


############
# PLOTTING #
############

hf.plot_excitations(momenta,Es; title="Energy levels MIL-53(V)")