var documenterSearchIndex = {"docs":
[{"location":"Examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Several examples can be found in the \"examples\" folder of the repository. In this tutorial, we elaborate on some of the details.","category":"page"},{"location":"Examples/#Initialization","page":"Examples","title":"Initialization","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"The first few lines of the script will often look very similar. We start by auto-activating the project \"Hubbard\" and enable local path handling from DrWatson","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"using DrWatson\n@quickactivate \"Hubbard\"","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Next, we import the required packages","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"using MPSKit\nusing KrylovKit","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"The command","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"include(projectdir(\"src\", \"HubbardFunctions.jl\"))\nimport .HubbardFunctions as hf","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"includes the module HubbardFunctions, independent of the current directory. This module stores all functions and structures. ","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Once everything is properly initialized, we proceed by defining the model of interest. We destinguish between one-band and multi-band models.","category":"page"},{"location":"Examples/#One-band","page":"Examples","title":"One-band","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Each Hubbard model is linked to a structure, storing all its major properties. The first question we ask ourselves is whether we want to conserve the number of electrons by imposing a U(1) symmetry, or if we want to add a chemical potential to deal with this. The first we achieve by using the OB_Sim() structure, the latter with OBC_Sim().","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"The most important properties of OB_Sim() are the hoppig t, the Hubbard u, the chemical potential µ, and the filling defined by the ratio P/Q.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"s = 2.5             # Schmidt cut value, determines bond dimension.\nP = 1;              # Filling of P/Q. P/Q = 1 is half-filling.\nQ = 1;\nbond_dim = 20;      # Initial bond dimension of the state. Impact on result is small as DMRG modifies it.\n\n# Define hopping, direct interaction, and chemical potential.\nt=[1.0, 0.1];\nu=[8.0];\nμ=0.0;\n\n# Spin=false will use SU(2) spin symmetry, the exact spin configuration cannot be deduced.\nSpin = false\n\nmodel = hf.OB_Sim(t, u, μ, P, Q, s, bond_dim; spin=Spin);","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"t is a vector where the first element is the nearest neighbour hopping, the second the next-nearest neighbour hopping, and so on. Similarly, the first element of u is the on-site Coulomb repulsion and the second element the interaction between two neighbouring sites. ","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"The Schmidt cut s determines to which value the bond dimension is grown by the iDMRG2 algorithm, while bond_dim is the maximal value used for the initialization of the MPS.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"NOTE: In order to preserve injectivity, a unit cell of size Q is used if Pis even and of size 2*Q if P is odd. Therefore, filling ratios that deviate from half filling P=Q=1 tend to be more intensive.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Finally, the tag spin determines if spin up and down have to be treated independently. If spin=false, an additional SU(2) symmetry is imposed, reducing the local Hilbert space dimension to 3 and leading to a substantial speed up. However, no information about the spin of a state can be retrieved.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"OBC_Sim() works similarly. Now, we either provide a chemical potential or a filling.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"model_OBC_1 = hf.OBC_Sim(t, u, P/Q, s, bond_dim; mu=false)\nmodel_OBC_2 = hf.OBC_Sim(t, u, μ, s, bond_dim; mu=true)","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"If a filling is defined, the corresponding chemical potential is sought iteratively. Calculations without spin symmetry are not yet implemented.","category":"page"},{"location":"Examples/#Multi-band","page":"Examples","title":"Multi-band","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"In analogy with the one-band model, multi-band models can be constructed using MB_Sim() or MBC_Sim. For the one-band model, DrWatson is able to find a unique name for the model based on its parameters. This name is later used to retrieve earlier computed results. For multi-band models, the number of parameters is simply too large and we have to provide a unique name ourselves, like the name of the script for instance.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"name_jl = last(splitpath(Base.source_path()))\nname = first(split(name_jl,\".\"))","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Then, we insert the parameters in the form of Btimes B matrices, where B is the number of bands. For a 2-band model this looks as follows","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"s = 2.5             # Schmidt cut value, determines bond dimension.\nP = 1;              # Filling of P/Q. P/Q = 1 is half-filling.\nQ = 1;\nbond_dim = 20;      # Initial bond dimension of the state. Impact on result is small as DMRG modifies it.\n\n# Define hopping, direct and exchange interaction matrices.\nt_onsite = [0.000 3.803 -0.548 0.000; 3.803 0.000 2.977 -0.501];\nt_intersite = [0.000 3.803 -0.548 0.000; 3.803 0.000 2.977 -0.501];\nt = cat(t_onsite,t_intersite, dims=2);\nU = [10.317 6.264 0.000 0.000; 6.264 10.317 6.162 0.000];\nJ = [0.000 0.123 0.000 0.000; 0.123 0.000 0.113 0.000];\nU13 = zeros(2,2)\n\nmodel = hf.MB_Sim(t, U, J, U13, P, Q, s, bond_dim; code = name);","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Where the one-band model used vectors for t and u, the multi-band model concatenates matrices horizontally. In addition, the exchange J and U_ijjj parameters, with zeros on the diagonals as these are included in u, are implemented as well. Since those parameters are usually rather small, U13 is an optional argument. Furthermore, parameters of the form U_ijkk and U_ijkl can be implemented by providing dictionaries as kwargs tot the structure","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"U112 = Dict((1,1,2,3) => 0.0, (1,2,4,2) => 0.0) # and so on ...\nU1111 = Dict((1,2,3,4) => 0.0, (3,2,4,1) => 0.0) # ...\nmodel = hf.MB_Sim(t, U, J, U13, P, Q, s, bond_dim; code = name, U112=U112, U1111=U1111)","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"An index i larger than B corresponds to band i modulo B on site iB.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"NOTE: When the parameters are changed but you want to keep the name of the model the same, you should put force=true to overwrite the previous results, obtained with the old parameters. Be cautious for accidentally overwriting data that you want to keep.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"For a MBC_Sim structure, we would have","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"model_MBC = hf.MBC_Sim(t, u, J, s, bond_dim; code=name);","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"The chemical potential is included in the diagonal terms of t_onsite. Iterative determination of the chemical potential for a certain filling is not yet supported.","category":"page"},{"location":"Examples/#Ground-state","page":"Examples","title":"Ground state","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"The ground state of a model is computed (or loaded if it has been computed before) by the function produce_groundstate. We can then extract the ground state energy as the expectation value of the Hamiltonian with respect to the ground state.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"dictionary = hf.produce_groundstate(model);\nψ₀ = dictionary[\"groundstate\"];\nH = dictionary[\"ham\"];\nE0 = expectation_value(ψ₀, H);\nE = real(E0)./length(H);\nprintln(\"Groundstate energy: $E\")","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Other properties, such as the bond dimension, the electron density and spin density (if it was a calculation without SU(2) symmetry), can be calculated as well.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"println(\"Bond dimension: $(hf.dim_state(ψ₀))\")\nprintln(\"Electron density: $(hf.density_state(ψ₀))\")\nprintln(\"Spin density: $(hf.density_spin(ψ₀))\")","category":"page"},{"location":"Examples/#Excited-states","page":"Examples","title":"Excited states","text":"","category":"section"},{"location":"Examples/","page":"Examples","title":"Examples","text":"To compute quasiparticle excitations we have to choose the momentum, the number of excitations, and the symmetry sectors. ","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"resolution = 5;\nmomenta = range(0, π, resolution);\nnums = 1;\n\nexc = hf.produce_excitations(model, momenta, nums; charges=[0,0.0,0]);\nEs = exc[\"Es\"];\nprintln(\"Excitation energies: \")\nprintln(Es)","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"Excitations in the same sector as the ground state are found by defining charges as zeros. These charges refer to the difference with the ground state. Be aware that the meaning of the charges in this vector differ depending on the symmetries and thus on the type of model. OBC_Sim and MBC_Sim even have only two symmetries, and hence, two charges.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"For example, a spin symmetric, electron conserving model has symmetry mathbbZ_2times SU(2)times U(1). Sectors obtained by adding a particle or hole differ by charges = [1,1/2,+/-1]. These single-particle excitations allow for the calculation of the band gap.","category":"page"},{"location":"Examples/","page":"Examples","title":"Examples","text":"gap, k = hf.produce_bandgap(model)\nprintln(\"Band Gap for s=$s: $gap eV at momentum $k\")","category":"page"},{"location":"Functions/#Library-documentation","page":"Library","title":"Library documentation","text":"","category":"section"},{"location":"Functions/#Model-structures","page":"Library","title":"Model structures","text":"","category":"section"},{"location":"Functions/","page":"Library","title":"Library","text":"HubbardFunctions.OB_Sim\nHubbardFunctions.OBC_Sim\nHubbardFunctions.MB_Sim\nHubbardFunctions.MBC_Sim","category":"page"},{"location":"Functions/#Main.HubbardFunctions.OB_Sim","page":"Library","title":"Main.HubbardFunctions.OB_Sim","text":"OB_Sim(t::Vector{Float64}, u::Vector{Float64}, μ=0.0, J::Vector{Float64}, P=1, Q=1, svalue=2.0, bond_dim=50, period=0; kwargs...)\n\nConstruct a parameter set for a 1D one-band Hubbard model with a fixed number of particles.\n\nArguments\n\nt: Vector in which element n is the value of the hopping parameter of distance n. The first element is the nearest-neighbour hopping.\nu: Vector in which element n is the value of the Coulomb interaction with site at distance n-1. The first element is the on-site interaction.\nJ: Vector in which element n is the value of the exchange interaction with site at distance n. The first element is the nearest-neighbour exchange.\nµ: The chemical potential.\nP,Q: The ratio P/Q defines the number of electrons per site, which should be larger than 0 and smaller than 2.\nsvalue: The Schmidt truncation value, used to truncate in the iDMRG2 algorithm for the computation of the groundstate.\nbond_dim: The maximal bond dimension used to initialize the state.\nPeriod: Perform simulations on a helix with circumference Period. Value 0 corresponds to an infinite chain.\n\nPut the optional argument spin=true to perform spin-dependent calculations.\n\n\n\n\n\n","category":"type"},{"location":"Functions/#Main.HubbardFunctions.OBC_Sim","page":"Library","title":"Main.HubbardFunctions.OBC_Sim","text":"OBC_Sim(t::Vector{Float64}, u::Vector{Float64}, μf::Float64, svalue=2.0, bond_dim=50, period=0; mu=true, kwargs...)\n\nConstruct a parameter set for a 1D one-band Hubbard model with the number of particles determined by a chemical potential.\n\nArguments\n\nt: Vector in which element n is the value of the hopping parameter of distance n. The first element is the nearest-neighbour hopping.\nu: Vector in which element n is the value of the Coulomb interaction with site at distance n-1. The first element is the on-site interaction.\nµf: The chemical potential, if mu=true. Otherwise, the filling of the system. The chemical potential corresponding to the given filling is determined automatically.\nsvalue: The Schmidt truncation value, used to truncate in the iDMRG2 algorithm for the computation of the groundstate.\nbond_dim: The maximal bond dimension used to initialize the state.\nPeriod: Perform simulations on a helix with circumference Period. Value 0 corresponds to an infinite chain.\n\nSpin-dependent calculations are not yet implemented.\n\n\n\n\n\n","category":"type"},{"location":"Functions/#Main.HubbardFunctions.MB_Sim","page":"Library","title":"Main.HubbardFunctions.MB_Sim","text":"MB_Sim(t::Matrix{Float64}, u::Matrix{Float64}, J::Matrix{Float64}, U13::Matrix{Float64}, P=1, Q=1, svalue=2.0, bond_dim=50; kwargs...)\n\nConstruct a parameter set for a 1D B-band Hubbard model with a fixed number of particles.\n\nArguments\n\nt: Bx(nB) matrix in which element (ij) is the hopping parameter from band i to band j. The on-site, nearest neighbour, next-to-nearest neighbour... hopping matrices are concatenated horizontally.\nu: Bx(nB) matrix in which element (ij) is the Coulomb repulsion U_ij=U_iijj between band i and band j. The on-site, nearest neighbour, next-to-nearest neighbour... matrices are concatenated horizontally.\nJ: Bx(nB) matrix in which element (ij) is the exchange J_ij=U_ijji=U_ijij between band i and band j. The on-site, nearest neighbour, next-to-nearest neighbour... matrices are concatenated horizontally. The diagonal terms of the on-site matrix are ignored.\nU13: BxB matrix in which element (ij) is the parameter U_ijjj=U_jijj=U_jjij=U_jjji between band i and band j. Only on-site. The diagonal terms of the on-site matrix are ignored. This argument is optional.\nP,Q: The ratio P/Q defines the number of electrons per site, which should be larger than 0 and smaller than 2.\nsvalue: The Schmidt truncation value, used to truncate in the iDMRG2 algorithm for the computation of the groundstate.\nbond_dim: The maximal bond dimension used to initialize the state.\n\nPut the optional argument 'spin=true' to perform spin-dependent calculations. \n\nU13 inter-site, Uijkk, and Uijkl can be inserted using kwargs.\n\nUse the optional argument name to assign a name to the model.  This is used to destinguish between different parameter sets: Wrong results could be loaded or overwritten if not used consistently!!!\n\n\n\n\n\n","category":"type"},{"location":"Functions/#Main.HubbardFunctions.MBC_Sim","page":"Library","title":"Main.HubbardFunctions.MBC_Sim","text":"MBC_Sim(t::Matrix{Float64}, u::Matrix{Float64}, J::Matrix{Float64}, U13::Matrix{Float64}, svalue=2.0, bond_dim=50; kwargs...)\n\nConstruct a parameter set for a 1D B-band Hubbard model with the number of particles determined by a chemical potential.\n\nArguments\n\nt: Btimes nB matrix in which element (ij) is the hopping parameter from band i to band j. The on-site, nearest neighbour, next-to-nearest neighbour... hopping matrices are concatenated horizontally. The diagonal terms of the on-site matrix determine the filling.\nu: Btimes nB matrix in which element (ij) is the Coulomb repulsion U_ij=U_iijj between band i and band j. The on-site, nearest neighbour, next-to-nearest neighbour... matrices are concatenated horizontally.\nJ: Btimes nB matrix in which element (ij) is the exchange J_ij=U_ijji=U_ijij between band i and band j. The on-site, nearest neighbour, next-to-nearest neighbour... matrices are concatenated horizontally. The diagonal terms of the on-site matrix are ignored.\nU13: Btimes B matrix in which element (ij) is the parameter U_ijjj=U_jijj=U_jjij=U_jjji between band i and band j. Only on-site. The diagonal terms of the on-site matrix are ignored. This argument is optional.\nsvalue: The Schmidt truncation value, used to truncate in the iDMRG2 algorithm for the computation of the groundstate.\nbond_dim: The maximal bond dimension used to initialize the state.\n\nSpin-dependent calculations are not yet implemented.\n\nU13 inter-site, Uijkk, and Uijkl can be inserted using kwargs.\n\nUse the optional argument name to assign a name to the model.  This is used to destinguish between different parameter sets: Wrong results could be loaded or overwritten if not used consistently!!!\n\n\n\n\n\n","category":"type"},{"location":"Functions/#Compute-states","page":"Library","title":"Compute states","text":"","category":"section"},{"location":"Functions/","page":"Library","title":"Library","text":"HubbardFunctions.produce_groundstate\nHubbardFunctions.produce_excitations\nHubbardFunctions.produce_bandgap\nHubbardFunctions.produce_TruncState","category":"page"},{"location":"Functions/#Main.HubbardFunctions.produce_groundstate","page":"Library","title":"Main.HubbardFunctions.produce_groundstate","text":"produce_groundstate(model::Simulation; force::Bool=false)\n\nCompute or load groundstate of the model. If force=true, overwrite existing calculation.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Main.HubbardFunctions.produce_excitations","page":"Library","title":"Main.HubbardFunctions.produce_excitations","text":"produce_excitations(model::Simulation, momenta, nums::Int64; force::Bool=false, charges::Vector{Float64}=[0,0.0,0], kwargs...)\n\nCompute or load quasiparticle excitations of the desired model.\n\nArguments\n\nmodel: Model for which excitations are sought.\nmomenta: Momenta of the quasiparticle excitations.\nnums: Number of excitations.\nforce: If true, overwrite existing calculation.\ncharges: Charges of the symmetry sector of the excitations.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Main.HubbardFunctions.produce_bandgap","page":"Library","title":"Main.HubbardFunctions.produce_bandgap","text":"produce_bandgap(model::Union{OB_Sim, MB_Sim}; resolution::Int64=5, force::Bool=false)\n\nCompute or load the band gap of the desired model.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Main.HubbardFunctions.produce_TruncState","page":"Library","title":"Main.HubbardFunctions.produce_TruncState","text":"produce_truncstate(model::Simulation, trunc_dim::Int64; trunc_scheme::Int64=0, force::Bool=false)\n\nCompute or load a truncated approximation of the groundstate.\n\nArguments\n\nmodel: Model for which the groundstate is to be truncated.\ntrunc_dim: Maximal bond dimension of the truncated state.\ntrunc_scheme: Scheme to perform the truncation. 0 = VUMPSSvdCut. 1 = SvdCut.\nforce: If true, overwrite existing calculation.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Properties-of-groundstate","page":"Library","title":"Properties of groundstate","text":"","category":"section"},{"location":"Functions/","page":"Library","title":"Library","text":"HubbardFunctions.dim_state\nHubbardFunctions.density_state\nHubbardFunctions.density_spin","category":"page"},{"location":"Functions/#Main.HubbardFunctions.dim_state","page":"Library","title":"Main.HubbardFunctions.dim_state","text":"dim_state(ψ::InfiniteMPS)\n\nDetermine the bond dimensions in an infinite MPS.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Main.HubbardFunctions.density_state","page":"Library","title":"Main.HubbardFunctions.density_state","text":"density_state(model::Simulation)\n\nCompute the number of electrons per site in the unit cell for the ground state.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Main.HubbardFunctions.density_spin","page":"Library","title":"Main.HubbardFunctions.density_spin","text":"density_spin(model::Union{OB_Sim,MB_Sim})\n\nCompute the density of spin up and spin down per site in the unit cell for the ground state.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Tools","page":"Library","title":"Tools","text":"","category":"section"},{"location":"Functions/","page":"Library","title":"Library","text":"HubbardFunctions.plot_excitations\nHubbardFunctions.plot_spin\nHubbardFunctions.extract_params","category":"page"},{"location":"Functions/#Main.HubbardFunctions.plot_excitations","page":"Library","title":"Main.HubbardFunctions.plot_excitations","text":"plot_excitations(momenta, energies; title=\"Excitation_energies\", l_margin=[15mm 0mm])\n\nPlot the obtained energy levels in functions of the momentum.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Main.HubbardFunctions.plot_spin","page":"Library","title":"Main.HubbardFunctions.plot_spin","text":"plot_spin(model::Simulation; title=\"Spin Density\", l_margin=[15mm 0mm])\n\nPlot the spin density of the model throughout the unit cell as a heatmap.\n\n\n\n\n\n","category":"function"},{"location":"Functions/#Main.HubbardFunctions.extract_params","page":"Library","title":"Main.HubbardFunctions.extract_params","text":"extract_params(path::String; range_u::Int64= 1, range_t::Int64=2, range_J::Int64=1, \n                    range_U13::Int64=1, r_1111::Int64 = 1, r_112::Int64 = 1)\n\nExtract the parameters from a params.jl file located at path in PyFoldHub format.\n\n\n\n\n\n","category":"function"},{"location":"#HubbardTN","page":"Home","title":"HubbardTN","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Welcome to the documentation for HubbardTN, a tool for implementing and solving general 1D multi-band Hubbard models using tensor networks.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To reproduce this project, do the following:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Download this code base.\nOpen a Julia console and do:\njulia> using Pkg\njulia> Pkg.add(\"DrWatson\") # install globally, for using `quickactivate`\njulia> Pkg.activate(\"path/to/this/project\")\njulia> Pkg.instantiate()","category":"page"},{"location":"","page":"Home","title":"Home","text":"This will install all necessary packages for you to be able to run the scripts and everything should work out of the box, including correctly finding local paths.","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The overall outline of the simulations is always the same. First, choose the type of Hubbard model that you are interested in. The different options are defined by their symmetries: ","category":"page"},{"location":"","page":"Home","title":"Home","text":"One-band model\nSpin symmetry and conserved number of electrons, mathbbZ_2times SU(2)times U(1).\nConserved number of spin up and down electrons, mathbbZ_2times U(1)times U(1).\nSpin symmetry, mathbbZ_2times SU(2).\nMulti-band model\nSpin symmetry and conserved number of electrons, mathbbZ_2times SU(2)times U(1).\nConserved number of spin up and down electrons, mathbbZ_2times U(1)times U(1).\nSpin symmetry, mathbbZ_2times SU(2).","category":"page"},{"location":"","page":"Home","title":"Home","text":"Proceed by inserting the parameters of the model. You are now ready to compute the ground and excited states and their properties!","category":"page"}]
}
