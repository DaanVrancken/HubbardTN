module Functions_ob_chem

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

struct Hubbard_Chem_Simulation <: Simulation
    t::Vector{Float64}
    u::Vector{Float64}
    μ::Union{Float64, Nothing}
    P::Union{Int64, Nothing}
    Q::Union{Int64, Nothing}
    svalue::Float64
    bond_dim::Int64
    period::Int64
    kwargs
    function Hubbard_Chem_Simulation(t, u, μ::Float64, svalue=2.0, bond_dim = 50, period = 0; kwargs...)
        return new(t, u, μ, nothing, nothing, svalue, bond_dim, period, kwargs)
    end
    function Hubbard_Chem_Simulation(t, u, P::Int64, Q::Int64, svalue=2.0, bond_dim = 50, period = 0; kwargs...)
        return new(t, u, nothing, P, Q, svalue, bond_dim, period, kwargs)
    end
end
name(::Hubbard_Chem_Simulation) = "hubbard_ob_chem"


function Base.string(s::TensorKit.ProductSector{Tuple{FermionParity,SU2Irrep}})
    return "Irrep[fℤ₂×SU₂]($(s.sectors[1].sector.n), $(s.sectors[2].j))"
end


###############
# Groundstate #
###############

function hamiltonian_sb(t::Vector{Float64}, u::Vector{Float64}, μ::Float64, L::Int64)
    D_hop = length(t)
    D_int = length(u)
    
    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)
    Vs = Vect[I]((1, 1 / 2) => 1)

    c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
    blocks(c⁺)[I((1, 1 // 2))] = [1.0+0.0im 0.0+0.0im]
    blocks(c⁺)[I((0, 0))] = [0.0+0.0im; sqrt(2)+0.0im;;]
    c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
    blocks(c)[I((1, 1 // 2))] = [1.0+0.0im; 0.0+0.0im;;]
    blocks(c)[I((0, 0))] = [0.0+0.0im sqrt(2)+0.0im]
    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 2.0]
    blocks(n)[I((1, 1 // 2))] .= 1.0+0.0im

    @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]
    twosite = twosite + twosite'
    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]

    onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(onesite)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 u[1]-2*μ] 
    blocks(onesite)[I((1, 1 // 2))] .= -μ
    
    H = @mpoham sum(onesite{i} for i in vertices(InfiniteChain(1)))
    if L == 0
        for range_hop in 1:D_hop
            h = @mpoham sum(-t[range_hop]*twosite{i,i+range_hop} for i in vertices(InfiniteChain(1)))
            H += h
        end
        for range_int in 2:D_int
            h = @mpoham sum(u[range_int]*nn{i,i+range_int} for i in vertices(InfiniteChain(1)))
            H += h
        end
    elseif D_hop==1 && D_int==1
        h = @mpoham sum(-t[1]*twosite{i,i+1} -t[1]*twosite{i,i+L} for i in vertices(InfiniteChain(1)))
        H += h
    else
        return error("No extended models in 2D.")
    end

    return H
end

function initialize_mps(operator, max_dimension::Int64)
    Ps = operator.pspaces
    V_right = accumulate(fuse, Ps)
    
    V_l = accumulate(fuse, dual.(Ps); init=one(first(Ps)))
    V_left = reverse(V_l)
    len = length(V_left)
    step = length(V_left)-1
    V_left = [view(V_left,len-step+1:len); view(V_left,1:len-step)]   # same as circshift(V_left,1)

    V = TensorKit.infimum.(V_left, V_right)

    Vmax = Vect[(FermionParity ⊠ Irrep[SU₂])]((0,0)=>1)     # find maximal virtual space

    for i in 0:1
        for j in 0:1//2:3//2
            Vmax = Vect[(FermionParity ⊠ Irrep[SU₂])]((i,j)=>max_dimension) ⊕ Vmax
        end
    end

    V_max = copy(V)      # if no copy(), V will change along when V_max is changed

    for i in 1:length(V_right)
        V_max[i] = Vmax
    end

    V_trunc = TensorKit.infimum.(V,V_max)

    return InfiniteMPS(Ps, V_trunc)
end

function compute_groundstate(simul::Hubbard_Chem_Simulation, μ::Float64)
    kwargs = simul.kwargs
    tol = get(kwargs, :tol, 1e-10)
    verbosity = get(kwargs, :verbosity, 0)
    maxiter = get(kwargs, :maxiter, Int(1e3))
    schmidtcut = 10.0^(-simul.svalue)

    H = hamiltonian_sb(simul.t,simul.u,μ,simul.period)
    ψ₀ = initialize_mps(H,simul.bond_dim)
    if length(H) > 1
        ψ₀, envs, = find_groundstate(ψ₀, H, IDMRG2(; trscheme=truncbelow(schmidtcut), tol=max(tol, schmidtcut/10), verbosity=verbosity))
    else
        ψ₀, envs, = find_groundstate(ψ₀, H, VUMPS(; tol=max(tol, schmidtcut/10), verbosity=verbosity))
        ψ₀ = changebonds(ψ₀, SvdCut(; trscheme=truncbelow(schmidtcut)))
        χ = sum(i -> dim(left_virtualspace(ψ₀, i)), 1:length(H))
        for i in 1:maxiter
            ψ₀, envs = changebonds(ψ₀, H, VUMPSSvdCut(;trscheme=truncbelow(schmidtcut)))
            ψ₀, = find_groundstate(ψ₀, H, VUMPS(; tol=max(tol, schmidtcut / 10), verbosity=verbosity), envs)
            ψ₀ = changebonds(ψ₀, SvdCut(; trscheme=truncbelow(schmidtcut)))
            χ′ = sum(i -> dim(left_virtualspace(ψ₀, i)), 1:length(H))
            isapprox(χ, χ′; rtol=0.05) && break
            χ = χ′
        end
    end

    alg = VUMPS(; maxiter=maxiter, tol=1e-5, verbosity=verbosity) &
        GradientGrassmann(; maxiter=maxiter, tol=tol, verbosity=verbosity)
    ψ, envs, δ = find_groundstate(ψ₀, H, alg)
    
    return Dict("groundstate" => ψ, "environments" => envs, "ham" => H, "delta" => δ, "config" => simul)
end

function compute_groundstate(simul::Hubbard_Chem_Simulation)
    verbosity_mu = get(simul.kwargs, :verbosity_mu, 0)   

    if simul.μ !== nothing
        dictionary = compute_groundstate(simul, simul.μ);
    else 
        P = simul.P
        Q = simul.Q
        tol_mu = get(simul.kwargs, :tol_mu, 1e-8)
        maxiter_mu = get(simul.kwargs, :maxiter_mu, 20)
        step_size = get(simul.kwargs, :step_size, 1.0)
        flag = false

        lower_bound = get(simul.kwargs, :lower_mu, 0.0)
        upper_bound = get(simul.kwargs, :upper_mu, 0.0)
        mid_point = (lower_bound + upper_bound)/2
        i = 1

        dictionary_l = compute_groundstate(simul, lower_bound);
        dictionary_u = deepcopy(dictionary_l)
        dictionary_sp = deepcopy(dictionary_l)
        while i<=maxiter_mu
            if abs(density_state(dictionary_u["groundstate"]) - P/Q) < tol_mu
                flag=true
                dictionary_sp = deepcopy(dictionary_u)
                mid_point = upper_bound
                break
            elseif abs(density_state(dictionary_l["groundstate"]) - P/Q) < tol_mu
                flag=true
                dictionary_sp = deepcopy(dictionary_l)
                mid_point = lower_bound
                break
            elseif density_state(dictionary_u["groundstate"]) < P/Q
                lower_bound = copy(upper_bound)
                upper_bound += step_size
                dictionary_u = compute_groundstate(simul, upper_bound)
            elseif density_state(dictionary_l["groundstate"]) > P/Q
                upper_bound = copy(lower_bound)
                lower_bound -= step_size
                dictionary_l = compute_groundstate(simul, lower_bound)
            else
                break
            end
            verbosity_mu==1 && @info "Iteration μ: $i => Lower bound: $lower_bound; Upper bound: $upper_bound"
            i+=1
        end
        if upper_bound>0.0
            value = "larger"
            dictionary = dictionary_u
        else
            value = "smaller"
            dictionary = dictionary_l
        end
        if i>maxiter_mu
            max_value = (i-1)*step_size
            @warn "The chemical potential is $value than: $max_value. Increase the stepsize."
        end

        while abs(density_state(dictionary["groundstate"]) - P/Q)>tol_mu && i<=maxiter_mu && !flag
            mid_point = (lower_bound + upper_bound)/2
            dictionary = compute_groundstate(simul, mid_point)
            if density_state(dictionary["groundstate"]) < P/Q
                lower_bound = copy(mid_point)
            else
                upper_bound = copy(mid_point)
            end
            verbosity_mu==1 && @info "Iteration μ: $i => Lower bound: $lower_bound; Upper bound: $upper_bound"
            i+=1
        end
        if i>maxiter_mu
            @warn "The chemical potential lies between $lower_bound and $upper_bound, but did not converge within the tolerance. Increase maxiter_mu."
        else
            verbosity_mu==1 && @info "Final chemical potential = $mid_point"
        end

        if flag
            dictionary = dictionary_sp
        end

        dictionary["μ"] = mid_point
    end

    return dictionary
end

function produce_groundstate(simul::Hubbard_Chem_Simulation; force=false)
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

function compute_excitations(simul::Hubbard_Chem_Simulation, momenta, nums::Int64; tol=1e-10, solver=GMRES())
    sector = fℤ₂(1) ⊠ SU2Irrep(1 // 2)
    dictionary = produce_groundstate(simul)
    ψ = dictionary["groundstate"]
    H = dictionary["ham"]
    envs = dictionary["environments"]
    Es, qps = excitations(H, QuasiparticleAnsatz(; tol=tol), momenta./length(ψ), ψ, envs; num=nums, sector=sector, solver=solver)
    return Dict("Es" => Es, "qps" => qps, "momenta" => momenta)
end

function produce_excitations(simul::Hubbard_Chem_Simulation, momenta, nums::Int64; force=false, tol=1e-10, solver=GMRES())
    t = simul.t 
    u = simul.u
    S = "excitations_nums=$nums"*"_t$t"*"_u$u"
    S = replace(S, ", " => "_")
    data, _ = produce_or_load(simul, datadir("sims", name(simul)); prefix=S, force=force) do cfg
        return compute_excitations(cfg, momenta, nums; tol=tol, solver=solver)
    end
    return data
end


####################
# State properties #
####################

function density_state(ψ)
    I = fℤ₂ ⊠ SU2Irrep
    Ps = physicalspace(ψ, 1)

    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 2.0]
    blocks(n)[I((1, 1 // 2))] .= 1.0+0.0im

    nₑ = @mpoham sum(n{i} for i in vertices(InfiniteStrip(1)))
    Nₑ = real(expectation_value(ψ, nₑ));

    return sum(Nₑ)
end

function dim_state(ψ)
    dimension = zeros(length(ψ))
    for i in 1:length(ψ)
        dimension[i] = dim(space(ψ.AL[i],1))
    end
    return dimension
end


end