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


############################
# DEPENDENCE ON PARAMETERS #
############################

U_max = 5;

u_range = 0:1:U_max;
E = zeros(U_max+1,1);
P = 1;
Q = 1;

for u in u_range
    model = hf.OB_Sim([1.0], [u], 0.0, P, Q, 2.0);
    dictionary = hf.produce_groundstate(model; force=true);
    ψ₀ = dictionary["groundstate"];
    H = dictionary["ham"];
    E0 = expectation_value(ψ₀, H);
    
    E[u+1] = sum(real(E0))/length(ψ₀);
end

plot(u_range,E,seriestype=:scatter,label="Numerical",minorgrid=true, title="Energy density for f=$P/$Q")
xlabel!("U/t")
ylabel!("Groundstate energy density")


#########################
# DEPENDENCE ON FILLING #
#########################

t = [1];
u = [4];
P = [1, 1, 3];
Q = [2, 1, 2];
E = zeros(length(P),1);

for i in eachindex(P)
    model = fo.Hubbard_Simulation(t, u, 0.0, P[i], Q[i], 2.0; verbosity=0);
    dictionary = fo.produce_groundstate(model; force=true);
    ψ₀ = dictionary["groundstate"];
    H = dictionary["ham"];
    E0 = expectation_value(ψ₀, H);
    
    E[i] = sum(real(E0))/length(ψ₀);
end

plot(P./Q,E,seriestype=:scatter,label="Numerical",minorgrid=true, title="Energy density for U/t = $u/$t")
xlabel!("f")
ylabel!("Groundstate energy density")


########################
# COMPUTE EXCTITATIONS #
########################

P=1;
Q=1;
u=[8.0];
t=[1.0, 0.25];

model = hf.OB_Sim(t, u, 0.0, P, Q, 2.0);

dictionary = hf.produce_groundstate(model, force=true);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))/length(H)
println("Groundstate energy: $E")

# E≈-0.3254136421761771

nums = 2;
resolution = 8;
momenta = range(0, π, resolution);

exc = hf.produce_excitations(model, momenta, nums; force=true);
Es = exc["Es"];

# Es≈[-0.7204048629748429 -0.5112043615623657 -0.38639163676016813; 
# -0.743423937569374 -0.5247194228281297 -0.3946759670712312; 
# -0.8121804521131547 -0.5620529406249675 -0.4182613655203239; 
# -0.9262565329008478 -0.6167493302670978 -0.5275892550188661; 
# -1.0855832832735741 -0.8635216511608923 -0.6701224923150702; 
# -1.303682579926918 -1.1686521145493032 -0.7452318362883539; 
# -1.5366008101352848 -1.3289089250382224 -0.835676312990831; 
# -1.629983390404155 -1.3823860159390946 -0.8664705331279315]


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

hf.plot_excitations(momenta,Es; title="Energy levels")