module Functions_oneband

using MKL 
#using ThreadPinning
using LinearAlgebra
using MPSKit, MPSKitModels
using TensorKit
using KrylovKit
using DataFrames
using DrWatson
using Plots, StatsPlots
using QuadGK, SpecialFunctions
using TensorOperations
using Revise


#include("kitaev.jl")

#=
function __init__()
    ThreadPinning.mkl_set_dynamic(0)
    BLAS.set_num_threads(1)
    pinthreads(:affinitymask)
    TensorOperations.disable_cache()
end
=#

abstract type Simulation end
name(s::Simulation) = string(typeof(s))

struct Hubbard_Simulation <: Simulation
    t::Vector{Float64}
    u::Vector{Float64}
    μ::Float64
    P::Int64
    Q::Int64
    svalue::Float64
    bond_dim::Int64
    period::Int64
    kwargs
    function Hubbard_Simulation(t, u, μ=0.0, P=1, Q=1, svalue=2.0, bond_dim = 50, period = 0; kwargs...)
        return new(t, u, μ, P, Q, svalue, bond_dim, period, kwargs)
    end
end
name(::Hubbard_Simulation) = "hubbard_sb"

function Base.string(s::TensorKit.ProductSector{Tuple{FermionParity,SU2Irrep,U1Irrep}})
    parts = map(x -> sprint(show, x; context=:typeinfo => typeof(x)), s.sectors)
    return "[fℤ₂×SU₂×U₁]$(parts)"
end


###############
# Groundstate #
###############

function hamiltonian_sb(simul::Hubbard_Simulation)
    t = simul.t
    u = simul.u
    μ = simul.μ
    P = simul.P
    Q = simul.Q
    L = simul.period

    D_hop = length(t)
    D_int = length(u)
    
    if iseven(P)
        T = Q
    else 
        T = 2*Q
    end
    
    I = fℤ₂ ⊠ SU2Irrep ⊠ U1Irrep
    Ps = Vect[I]((0, 0, -P) => 1, (0, 0, 2*Q-P) => 1, (1, 1 // 2, Q-P) => 1)
    Vs = Vect[I]((1, 1 / 2, Q) => 1)

    c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
    blocks(c⁺)[I((1, 1 // 2, Q-P))] .= 1
    blocks(c⁺)[I((0, 0, 2*Q-P))] .= sqrt(2)
    c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
    blocks(c)[I((1, 1 / 2, Q-P))] .= 1
    blocks(c)[I((0, 0, 2*Q-P))] .= sqrt(2)
    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0, 2*Q-P))] .= 2
    blocks(n)[I((1, 1 // 2, Q-P))] .= 1

    @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]
    twosite = twosite + twosite'
    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]

    onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(onesite)[I((0, 0, 2*Q-P))] .= u[1] - 2*μ 
    blocks(onesite)[I((1, 1 // 2, Q-P))] .= -μ
    
    H = @mpoham sum(onesite{i} for i in vertices(InfiniteChain(T)))
    if L == 0
        for range_hop in 1:D_hop
            h = @mpoham sum(-t[range_hop]*twosite{i,i+range_hop} for i in vertices(InfiniteChain(T)))
            H += h
        end
        for range_int in 2:D_int
            h = @mpoham sum(u[range_int]*nn{i,i+range_int} for i in vertices(InfiniteChain(T)))
            H += h
        end
    elseif D_hop==1 && D_int==1
        h = @mpoham sum(-t[1]*twosite{i,i+1} -t[1]*twosite{i,i+L} for i in vertices(InfiniteChain(T)))
        H += h
    else
        return error("No extended models in 2D.")
    end

    return H
end

function initialize_mps(operator, P, max_dimension::Int64)
    Ps = operator.pspaces
    T = length(Ps)                      # value of Q or 2Q
    V_right = accumulate(fuse, Ps)
    
    V_l = accumulate(fuse, dual.(Ps); init=one(first(Ps)))
    V_left = reverse(V_l)
    len = length(V_left)
    step = length(V_left)-1
    V_left = [view(V_left,len-step+1:len); view(V_left,1:len-step)]   # same as circshift(V_left,1)

    V = TensorKit.infimum.(V_left, V_right)

    Vmax = Vect[(FermionParity ⊠ Irrep[SU₂] ⊠ Irrep[U₁])]((0,0,0)=>1)     # find maximal virtual space

    for i in 0:1
        for j in 0:1//2:3//2
            for k in -(P*T):1:(P*T)
                Vmax = Vect[(FermionParity ⊠ Irrep[SU₂] ⊠ Irrep[U₁])]((i,j,k)=>max_dimension) ⊕ Vmax
            end
        end
    end

    V_max = copy(V)      # if no copy(), V will change along when V_max is changed

    for i in 1:length(V_right)
        V_max[i] = Vmax
    end

    V_trunc = TensorKit.infimum.(V,V_max)

    return InfiniteMPS(Ps, V_trunc)
end

function compute_groundstate(simul::Hubbard_Simulation)
    H = hamiltonian_sb(simul)
    ψ₀ = initialize_mps(H,simul.P,simul.bond_dim)
    
    kwargs = simul.kwargs
    
    tol = get(kwargs, :tol, 1e-10)
    verbosity = get(kwargs, :verbosity, 3)
    maxiter = get(kwargs, :maxiter, Int(1e3))
    
    schmidtcut = 10.0^(-simul.svalue)
    
    if length(H) > 1
        ψ₀, envs, = find_groundstate(ψ₀, H, IDMRG2(; trscheme=truncbelow(schmidtcut), tol=max(tol, schmidtcut/10), verbosity=verbosity))
    else
        error("Hamiltonian has length 1. Unit cell is too small. Filling 2 is not supported.")
        # https://github.com/lkdvos/Hubbard/blob/e6aa3f39871a8d128019ce101644798db19082dc/src/Hubbard.jl
    end
    
    alg = VUMPS(; maxiter=maxiter, tol=1e-5, verbosity=verbosity) &
        GradientGrassmann(; maxiter=maxiter, tol=tol, verbosity=verbosity)
    ψ, envs, δ = find_groundstate(ψ₀, H, alg)
    
    #p = entanglementplot(ψ)
    #savefig(p, plotsdir(name(simul), savename("entanglement", simul, "png")))
    
    return Dict("groundstate" => ψ, "environments" => envs, "ham" => H, "delta" => δ, "config" => simul)
end

function produce_groundstate(simul::Hubbard_Simulation; force=false)
    t = simul.t 
    u = simul.u
    S = "groundstate_"*"t$t"*"_u$u"
    S = replace(S, ", " => "_")
    data, _ = produce_or_load(compute_groundstate, simul, datadir("sims", name(simul)); prefix=S, force=force)
    return data
end


###############
# Excitations #
###############
 
# would be nice if we could load first excitations and only calculate the higher excitations that were not produced yet
# the function "excitations" does not support this I think...

function compute_excitations_pos(simul::Hubbard_Simulation, momenta, nums::Int64; tol=1e-10, solver=GMRES())
    Q = simul.Q
    sector = fℤ₂(1) ⊠ SU2Irrep(1 // 2) ⊠ U1Irrep(Q)
    dictionary = produce_groundstate(simul)
    ψ = dictionary["groundstate"]
    H = dictionary["ham"]
    envs = dictionary["environments"]
    Es, qps = excitations(H, QuasiparticleAnsatz(; tol=tol), momenta./length(H), ψ, envs; num=nums, sector=sector, solver=solver)
    return Dict("Es_pos" => Es, "qps_pos" => qps, "momenta" => momenta)
end

function produce_excitations_pos(simul::Hubbard_Simulation, momenta, nums::Int64; force=false, tol=1e-10, solver=GMRES())
    t = simul.t 
    u = simul.u
    S = "excitations_pos_nums=$nums"*"_t$t"*"_u$u"
    S = replace(S, ", " => "_")
    data, _ = produce_or_load(simul, datadir("sims", name(simul)); prefix=S, force=force) do cfg
        return compute_excitations_pos(cfg, momenta, nums; tol=tol, solver=solver)
    end
    return data
end

function compute_excitations_neg(simul::Hubbard_Simulation, momenta, nums::Int64; tol=1e-10, solver=GMRES())
    Q = simul.Q
    sector = fℤ₂(1) ⊠ SU2Irrep(1 // 2) ⊠ U1Irrep(-Q)
    dictionary = produce_groundstate(simul)
    ψ = dictionary["groundstate"]
    H = dictionary["ham"]
    envs = dictionary["environments"]
    Es, qps = excitations(H, QuasiparticleAnsatz(; tol=tol), momenta./length(H), ψ, envs; num=nums, sector=sector, solver=solver)
    return Dict("Es_neg" => Es, "qps_neg" => qps, "momenta" => momenta)
end

function produce_excitations_neg(simul::Hubbard_Simulation, momenta, nums::Int64; force=false, tol=1e-10, solver=GMRES())
    t = simul.t 
    u = simul.u
    S = "excitations_neg_nums=$nums"*"_t$t"*"_u$u"
    S = replace(S, ", " => "_")
    data, _ = produce_or_load(simul, datadir("sims", name(simul)); prefix=S, force=force) do cfg
        return compute_excitations_neg(cfg, momenta, nums; tol=tol, solver=solver)
    end
    return data
end


####################
# State properties #
####################

function density_state(simul::Hubbard_Simulation)
    P = simul.P
    Q = simul.Q
    Bands = 1
    if iseven(P)
        T = Q
    else 
        T = 2*Q
    end

    dictionary = produce_groundstate(simul);
    ψ₀ = dictionary["groundstate"];
    I = fℤ₂ ⊠ SU2Irrep ⊠ U1Irrep
    Ps = physicalspace(ψ₀, 1)

    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0, 2*Q-P))] .= 2
    blocks(n)[I((1, 1 // 2, Q-P))] .= 1
    blocks(n)[I((0, 0, -P))] .= 0

    nₑ = @mpoham sum(n{i} for i in vertices(InfiniteStrip(Bands,T*Bands)))
    Nₑ = zeros(Bands*T,1);

    for i in 1:(Bands*T)
        Nₑ[i] = real(expectation_value(ψ₀, nₑ)[i])
    end
    
    N_av = zeros(Bands,1)
    for i in 1:Bands
        av = 0
        for j in 0:(T-1)
            av = Nₑ[i+Bands*j] + av
        end
        N_av[i] = av/T
    end

    check = (sum(Nₑ)/(T*Bands) ≈ P/Q)
    println("Filling is conserved: $check")

    return N_av
end

function density_state(ψ₀,P::Int64,Q::Int64)
    if iseven(P)
        T = Q
    else 
        T = 2*Q
    end
    Bands = 1

    I = fℤ₂ ⊠ SU2Irrep ⊠ U1Irrep
    Ps = physicalspace(ψ₀, 1)

    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0, 2*Q-P))] .= 2
    blocks(n)[I((1, 1 // 2, Q-P))] .= 1
    blocks(n)[I((0, 0, -P))] .= 0

    nₑ = @mpoham sum(n{i} for i in vertices(InfiniteStrip(Bands,T*Bands)))
    Nₑ = zeros(Bands*T,1);

    for i in 1:(Bands*T)
        Nₑ[i] = real(expectation_value(ψ₀, nₑ)[i])
    end
    
    N_av = zeros(Bands,1)
    for i in 1:Bands
        av = 0
        for j in 0:(T-1)
            av = Nₑ[i+Bands*j] + av
        end
        N_av[i] = av/T
    end

    check = (sum(Nₑ)/(T*Bands) ≈ P/Q)
    println("Filling is conserved: $check")

    return N_av
end

function dim_state(ψ)
    dimension = zeros(length(ψ))
    for i in 1:length(ψ)
        dimension[i] = dim(space(ψ.AL[i],1))
    end
    return dimension
end


end