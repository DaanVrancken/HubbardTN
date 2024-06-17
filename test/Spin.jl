println("
##########
#  Spin  #
##########
")

##################
# INITIALISATION #
##################

tol = 1e-3

model1 = hf.OB_Sim([1.0],[8.0], 0.0,1,1,2.0;spin=true)

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
model2 = hf.MB_Sim(t, u, J, P, Q, 2.0, bond_dim; code = name);


###############
# GROUNDSTATE #
###############

@testset "Groundstate" begin
    dictionary1 = hf.produce_groundstate(model1;force=true);
    ψ1 = dictionary1["groundstate"];
    H1 = dictionary1["ham"];
    E01 = expectation_value(ψ1, H1);
    E1 = sum(real(E01))./length(H1);
    println("Groundstate energy: $E1")

    dictionary2 = hf.produce_groundstate(model2;force=true);
    ψ2 = dictionary2["groundstate"];
    H2 = dictionary2["ham"];
    E02 = expectation_value(ψ2, H2);
    E2 = sum(real(E02))./length(H2);
    println("Groundstate energy: $E2")
end

###############
# EXCITATIONS #
###############

@testset "Excitations" begin 
    nums = 1;
    resolution = 5;
    momenta = range(0, π, resolution);

    exc = hf.produce_excitations(model1, momenta, nums; charges=[0,0.0,0]);
    Es = exc["Es"];
    @test real(Es)≈Es_norm atol=tol
    @test imag(Es)≈zeros(size(Es)) atol=1e-8
end


#########
# Tools #
#########

@testset "Tools" begin
    N1 = hf.density_state(model1)
    Nup1, Ndown1 = hf.density_spin(model1)
    println(Nup1)
    println(Ndown1)

    @test N1 ≈ 1.0

    N2 = hf.density_state(model2)
    Nup2, Ndown2 = hf.density_spin(model2)
    println(Nup2)
    println(Ndown2)

    @test N2 ≈ 1.0
end