println("
################
#  Multi-Band  #
################
")

##################
# INITIALISATION #
##################

tol = 1e-1

# Extract name of the current file. Will be used as code name for the simulation.
name_jl = last(splitpath(Base.source_path()))
name = first(split(name_jl,"."))


#################
# DEFINE SYSTEM #
#################

# Model for general tests
t_OS = [0.0 0.0; 0.0 0.0];
t_IS = [1.0 0.0; 0.0 1.0];
t = cat(t_OS,t_IS, dims=2)
U = [3.0 0.0; 0.0 3.0]
V = [0.0 0.0; 0.0 0.0]
u = cat(U,V,dims=2)
J = [0.0 0.0; 0.0 0.0]

P = 1;
Q = 1;
bond_dim = 20;

model = hf.MB_Sim(t, u, J, P, Q, 2.0, bond_dim; code = name);

# Model for testing nontrivial interactions
t_OS2 = [0.5 0.1; 0.1 0.5];
t_IS2 = [1.0 0.5; 0.5 1.0];
t2 = cat(t_OS2,t_IS2, dims=2)
U2 = [3.0 0.0; 0.0 3.0]
V2 = [0.25 0.0; 0.0 0.25]
u2 = cat(U2,V2,dims=2)
J2 = [0.0 0.5; 0.5 0.0]

model2 = hf.MB_Sim(t2, u2, J2, P, Q, 2.0, bond_dim; code = name*"2");


###############
# GROUNDSTATE #
###############

dictionary = hf.produce_groundstate(model; force=false);
dictionary2 = hf.produce_groundstate(model2; force=true);

@testset "Groundstate" begin
    E_norm = -0.630375296

    ψ₀ = dictionary["groundstate"];
    H = dictionary["ham"];
    E0 = expectation_value(ψ₀, H);
    E = sum(real(E0))/length(H)
    @test E≈E_norm atol=tol

    ψ2 = dictionary2["groundstate"];
    H2 = dictionary["ham"];
    E02 = expectation_value(ψ2, H2);
    E2 = sum(real(E02))/length(H2)
    @test imag(E2)≈0.0 atol=1e-8
end


###############
# EXCITATIONS #
###############

@testset "Excitations" begin
    resolution = 5;
    momenta = range(0, π, resolution);
    nums = 1;

    exc = hf.produce_excitations(model, momenta, nums; force=true, charges=[1,0.5,1]);
    Es = exc["Es"];
    @test imag(Es)≈zeros(size(Es)) atol=1e-8
end


#########
# Tools #
#########

@testset "Tools" begin
    D = hf.dim_state(dictionary["groundstate"])
    @test typeof(D) == Vector{Int64}
    @test D > zeros(size(D))

    electron_number = hf.density_state(model)
    @test sum(electron_number)/2 ≈ P/Q atol=1e-8
end