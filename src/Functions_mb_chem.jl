module Functions_mb_chem

using MKL 
using ThreadPinning
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

function __init__()
    LinearAlgebra.BLAS.set_num_threads(1)
    ThreadPinning.pinthreads(:cores)
end

abstract type Simulation end
name(s::Simulation) = string(typeof(s))

struct Hubbard_MBC_Simulation <: Simulation
    t::Matrix{Float64}                        #convention: number of bands = number of rows, BxB for on-site + BxB*range matrix for IS
    U::Matrix{Float64}                        #convention: BxB matrix for OS (with OB on diagonal) + BxB*range matrix for IS
    J::Matrix{Float64}                        #convention: BxB matrix for OS (with OB zeros) + BxB*range matrix for IS
    μ::Array{Float64, 1}                      #convention: Bx1 array
    svalue::Float64
    bond_dim::Int64
    kwargs
    function Hubbard_MBC_Simulation(t, u, J, μ, svalue=2.0, bond_dim = 50; kwargs...)
        return new(t, u, J, μ, svalue, bond_dim, kwargs)
    end
end
name(::Hubbard_MBC_Simulation) = "Hubbard_mb_chem"    #give a different name

function Base.string(s::TensorKit.ProductSector{Tuple{SU2Irrep,U1Irrep}})
    parts = map(x -> sprint(show, x; context=:typeinfo => typeof(x)), s.sectors)
    return "[fℤ₂×SU₂]$(parts)"
end


###############
# Hamiltonian #
###############

function hopping()   
    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)
    Vs = Vect[I]((1, 1 / 2) => 1)

    c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
    blocks(c⁺)[I((1, 1 // 2))] = [1.0+0.0im 0.0+0.0im]
    blocks(c⁺)[I((0, 0))] = [0.0+0.0im; sqrt(2)+0.0im;;]

    c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
    blocks(c)[I((1, 1 // 2))] = [1.0+0.0im; 0.0+0.0im;;]
    blocks(c)[I((0, 0))] = [0.0+0.0im sqrt(2)+0.0im]

    @planar twosite_hopping[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]
    
    return twosite_hopping, twosite_hopping'   # hopping is no longer necessarily the same in both directions
end

# t[i,j] gives the hopping of band i on one site to band j on the same site (i≠j)
function OS_hopping(t)
    Bands,Bands2 = size(t)
    
    if Bands ≠ Bands2 || typeof(t) ≠ Matrix{Float64}
        @warn "t is not a float square matrix."
    end
    diagonal = zeros(Bands,1)
    diagonal_zeros = zeros(Bands,1)
    for i in 1:Bands
        diagonal[i] = t[i,i]
    end
    if diagonal≠diagonal_zeros
        @warn "On-band hopping is not taken into account in OS_hopping."
    end
    
    cdc,_ = hopping()
    Lattice = InfiniteStrip(Bands,Bands)
        
    # Define necessary different indices of sites/orbitals in the lattice
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    return @mpoham sum(-t[bi,bf]*cdc{Lattice[bf,site],Lattice[bi,site]} for (site, bi, bf) in Indices)
end

# t[i,j] gives the hopping of band i on one site to band j on the range^th next site
# parameter must be equal in both directions (1i->2j=2j->1i) to guarantee hermiticity
# Translation invariance implies then that we should have t'=t (unless due to geometry like for 6-band model)
function IS_hopping(t,range)
    Bands,Bands2 = size(t)
    if Bands ≠ Bands2 || typeof(t) ≠ Matrix{Float64}
        @warn "t is not a float square matrix"
    end
    
    cdc, ccd = hopping()
    Lattice = InfiniteStrip(Bands,Bands)
        
    # Define necessary different indices of sites/orbitals in the lattice
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(Bands^2)]
    
    return @mpoham sum(-t[bi,bf]*(cdc{Lattice[bf,site+range],Lattice[bi,site]} + 
                        ccd{Lattice[bf,site+range],Lattice[bi,site]}) for (site, bi, bf) in Indices)
end

# μ[i] gives the hopping of band i on one site to band i on the same site, combine this one with OB_hopping?
function Chem_pot(μ)
    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)

    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 2.0]
    blocks(n)[I((1, 1 // 2))] .= 1.0+0.0im
    
    Bands = length(μ)
    
    Lattice = InfiniteStrip(Bands,Bands)
    
    Indices = [(div(l-1,Bands)+1, mod(l-1,Bands)+1) for l in 1:(Bands)]
    
    return @mpoham sum(-μ[i]*n{Lattice[i,j]} for (j,i) in Indices)
end

# u[i] gives the interaction on band i
function OB_interaction(u)
    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)
    
    Bands = length(u)

    onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(onesite)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 1.0] 
    Lattice = InfiniteStrip(Bands,Bands)
    
    Indices = [(div(l-1,Bands)+1, mod(l-1,Bands)+1) for l in 1:(Bands)]
    
    return @mpoham sum(u[i]*onesite{Lattice[i,j]} for (j,i) in Indices)
end

# U[i,j] gives the direct interaction between band i on one site to band j on the same site. Averaged over U[i,j] and U[j,i]
function Direct_OS(U)
    Bands,Bands2 = size(U)
    
    if Bands ≠ Bands2 || typeof(U) ≠ Matrix{Float64}
        @warn "U is not a float square matrix"
    end
    
    U_av = zeros(Bands,Bands2)
    for i in 2:Bands    
        for j in 1:(i-1)
            U_av[i,j] = 0.5*(U[i,j]+U[j,i])
        end
    end
    
    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)
    
    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 2.0]
    blocks(n)[I((1, 1 // 2))] .= 1.0+0.0im
    
    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]
    Lattice = InfiniteStrip(Bands,Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    return @mpoham sum(U_av[bi,bf]*nn{Lattice[bi,site],Lattice[bf,site]} for (site,bi,bf) in Indices if U_av[bi,bf]≠0.0)
end

# J[i,j] gives the exchange interaction between band i on one site to band j on the same site.
function Exchange1_OS(J)
    Bands,Bands2 = size(J)
    
    if Bands ≠ Bands2 || typeof(J) ≠ Matrix{Float64}
        @warn "J is not a float square matrix"
    end
    diagonal = zeros(Bands,1)
    diagonal_zeros = zeros(Bands,1)
    for i in 1:Bands
        diagonal[i] = J[i,i]
    end
    if diagonal≠diagonal_zeros
        @warn "On-band interaction is not taken into account in Exchange_OS."
    end
    
    cdc,_ = hopping()
    
    @tensor C4[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[-2 3; 2 -3]
    Lattice = InfiniteStrip(Bands,Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    return @mpoham sum(0.5*J[bi,bf]*C4{Lattice[bi,site],Lattice[bf,site]} for (site,bi,bf) in Indices)
end;

function Exchange2_OS(J)
    Bands,Bands2 = size(J)
    
    if Bands ≠ Bands2 || typeof(J) ≠ Matrix{Float64}
        @warn "J is not a float square matrix"
    end
    diagonal = zeros(Bands,1)
    diagonal_zeros = zeros(Bands,1)
    for i in 1:Bands
        diagonal[i] = J[i,i]
    end
    if diagonal≠diagonal_zeros
        @warn "On-band interaction is not taken into account in Exchange_OS."
    end
    
    cdc,_ = hopping()
    
    @tensor C4[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[3 -2; -3 2]
    Lattice = InfiniteStrip(Bands,Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    return @mpoham sum(0.5*J[bi,bf]*C4{Lattice[bi,site],Lattice[bf,site]} for (site,bi,bf) in Indices)
end;

# V[i,j] gives the direct interaction between band i on one site to band j on the range^th next site.
function Direct_IS(V,range)
    Bands,Bands2 = size(V)
    
    if Bands ≠ Bands2 || typeof(V) ≠ Matrix{Float64}
        @warn "V is not a float square matrix"
    end
    
    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)
    
    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 2.0]
    blocks(n)[I((1, 1 // 2))] .= 1.0+0.0im
    
    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]
    Lattice = InfiniteStrip(Bands,Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(Bands^2)]
    
    return @mpoham sum(V[bi,bf]*nn{Lattice[bi,site],Lattice[bf,site+range]} for (site,bi,bf) in Indices)
end

# J[i,j] gives the exchange interaction between band i on one site to band j on the range^th next site.
function Exchange1_IS(J,range)
    Bands,Bands2 = size(J)
    
    if Bands ≠ Bands2 || typeof(J) ≠ Matrix{Float64}
        @warn "J is not a float square matrix"
    end
    
    cdc,_ = hopping()
    
    @tensor C4[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[-2 3; 2 -3]
    Lattice = InfiniteStrip(Bands,Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(Bands^2)]
    
    return @mpoham sum(J[bi,bf]*C4{Lattice[bi,site],Lattice[bf,site+range]} for (site,bi,bf) in Indices)    # operator has no direction
end;

function Exchange2_IS(J,range)
    Bands,Bands2 = size(J)
    
    if Bands ≠ Bands2 || typeof(J) ≠ Matrix{Float64}
        @warn "J is not a float square matrix"
    end
    
    cdc,_ = hopping()
    
    @tensor C4[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[3 -2; -3 2]
    Lattice = InfiniteStrip(Bands,Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(Bands^2)]
    
    return @mpoham sum(0.5*J[bi,bf]*C4{Lattice[bi,site],Lattice[bf,site+range]} + 0.5*J[bi,bf]*C4{Lattice[bi,site+range],Lattice[bf,site]} for (site,bi,bf) in Indices) #operator has direction
end;

function hamiltonian_mb(simul::Hubbard_MBC_Simulation)
    t = simul.t
    u = simul.U
    J = simul.J
    μ = simul.μ

    Bands,width_t = size(t)
    Bands1,width_u = size(u)
    Bands2 = length(μ)
    Bands3, width_J = size(J)

    if !(Bands == Bands1 == Bands2 == Bands3)
        return error("Number of bands is incosistent.")
    end

    Range_t = Int((width_t-Bands)/Bands)
    Range_u = Int((width_u-Bands)/Bands)
    Range_J = Int((width_J-Bands)/Bands)

    u_OB = zeros(Bands)
    for i in 1:Bands
        u_OB[i] = u[i,i]
    end
    if u_OB == zeros(Bands)
        @warn "No on-band interaction found. This may lead to too low contributions of other Hamiltonian terms."
    end
    H_total = OB_interaction(u_OB)

    t_OS = t[:,1:Bands]
    if t_OS != zeros(Bands,Bands)
        H_total += OS_hopping(t_OS)
    end

    for i in 1:Range_t
        t_IS = t[:,(Bands*i+1):(Bands*(i+1))]
        if t_IS != zeros(Bands,Bands)
            H_total += IS_hopping(t_IS,i)
        end
    end

    if μ != zeros(Bands)
        H_total += Chem_pot(μ)
    end

    u_OS = u[:,1:Bands]
    for i in 1:Bands
        u_OS[i,i] = 0.0
    end
    if u_OS != zeros(Bands,Bands)
        H_total += Direct_OS(u_OS)
    end

    for i in 1:Range_u
        V = u[:,(Bands*i+1):(Bands*(i+1))]
        if V != zeros(Bands,Bands)
            H_total += Direct_IS(V,i)
        end
    end

    J_OS = J[:,1:Bands]
    if J_OS != zeros(Bands,Bands)
        H_total += Exchange1_OS(J_OS) + Exchange2_OS(J_OS)
    end

    for i in 1:Range_J
        J_IS = J[:,(Bands*i+1):(Bands*(i+1))]
        if J_IS != zeros(Bands,Bands)
            H_total += Exchange1_IS(J_IS,i) + Exchange2_IS(J_IS,i) + H_exch_IS
        end
    end

    return H_total
end


###############
# Groundstate #
###############

# Ps is now vector of length Bands
function initialize_mps(operator, max_dimension)
    Ps = operator.pspaces
    L = length(Ps)
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

function compute_groundstate(simul::Hubbard_MBC_Simulation)
    H = hamiltonian_mb(simul)
    ψ₀ = initialize_mps(H,simul.bond_dim)
    
    kwargs = simul.kwargs
    
    tol = get(kwargs, :tol, 1e-10)
    verbosity = get(kwargs, :verbosity, 3)
    maxiter = get(kwargs, :maxiter, Int(1e3))
    
    schmidtcut = 10.0^(-simul.svalue)
    
    if length(H) > 1
        ψ₀, envs, = find_groundstate(ψ₀, H, IDMRG2(; trscheme=truncbelow(schmidtcut), tol=max(tol, schmidtcut/10), verbosity=verbosity))
    else
        error("Hamiltonian has length 1. Unit cell is too small.")
        # https://github.com/lkdvos/Hubbard/blob/e6aa3f39871a8d128019ce101644798db19082dc/src/Hubbard.jl
    end
    
    alg = VUMPS(; maxiter=maxiter, tol=1e-5, verbosity=verbosity) &
        GradientGrassmann(; maxiter=maxiter, tol=tol, verbosity=verbosity)
    ψ, envs, δ = find_groundstate(ψ₀, H, alg)
    
    return Dict("groundstate" => ψ, "environments" => envs, "ham" => H, "delta" => δ, "config" => simul)
end

function produce_groundstate(simul::Hubbard_MBC_Simulation; force=false)
    band = length(simul.μ)
    code = get(simul.kwargs, :code, "bands=$band")
    data, _ = produce_or_load(compute_groundstate, simul, datadir("sims", name(simul)); prefix="groundstate_"*code, force=force)
    return data
end


###############
# Excitations #
###############

# would be nice if we could load first excitations and only calculate the higher excitations that were not produced yet
# Krylovkit does not support this yet

function compute_excitations(simul::Hubbard_MBC_Simulation, momenta, nums::Int64; tol=1e-10, trunc_dim::Int64=0, solver=GMRES())
    if trunc_dim<0
        return error("Trunc_dim should be a positive integer.")
    end

    sector = fℤ₂(1) ⊠ SU2Irrep(1 // 2)
    dictionary = produce_groundstate(simul)
    ψ = dictionary["groundstate"]
    H = dictionary["ham"]
    if trunc_dim==0
        envs = dictionary["environments"]
    else
        ψ, envs = changebonds(ψ,H,VUMPSSvdCut(; trscheme=truncdim(trunc_dim)))
    end
    Es, qps = excitations(H, QuasiparticleAnsatz(; tol=tol), momenta./length(H), ψ, envs; num=nums, sector=sector, solver=solver)
    return Dict("Es" => Es, "qps" => qps, "momenta" => momenta)
end

function produce_excitations(simul::Hubbard_MBC_Simulation, momenta, nums::Int64; force=false, tol=1e-10, trunc_dim::Int64=0, solver=GMRES())
    band = length(simul.μ)
    if typeof(momenta)==Float64
        momenta_string = "_mom=$momenta"
    else
        mom_1 = first(momenta)
        mom_last = last(momenta)
        mom_length = length(momenta)
        momenta_string = "_mom=$mom_1 to$mom_last div$mom_length"
        momenta_string = replace(momenta_string, " " => "" )
    end
    code = get(simul.kwargs, :code, "bands=$band")
    data, _ = produce_or_load(simul, datadir("sims", name(simul)); prefix="excitations_"*code*"_nums=$nums"*"_tol=$tol"*"_trunc=$trunc_dim"*momenta_string, force=force) do cfg
        return compute_excitations(cfg, momenta, nums; tol=tol, trunc_dim=trunc_dim, solver=solver)
    end
    return data
end


####################
# State properties #
####################

function density_state(ψ₀)
    Bands = length(ψ₀)

    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)

    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 2.0]
    blocks(n)[I((1, 1 // 2))] .= 1.0+0.0im

    nₑ = @mpoham sum(n{i} for i in vertices(InfiniteStrip(Bands,Bands)))
    Nₑ = zeros(Bands,1);

    for i in 1:(Bands)
        Nₑ[i] = real(expectation_value(ψ₀, nₑ)[i])
    end

    return Nₑ
end

function dim_state(ψ)
    dimension = zeros(length(ψ))
    for i in 1:length(ψ)
        dimension[i] = dim(space(ψ.AL[i],1))
    end
    return dimension
end

        
end
