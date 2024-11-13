# Examples
Several examples can be found in the "examples" folder of the repository. In this tutorial, we elaborate on some of the details.

## Initialization
The first few lines of the script will often look very similar. We start by importing the required packages.
```
using MPSKit, KrylovKit
using Printf, Plots, Revise, DrWatson
```
Next, we auto-activate the project and enable local path handling from DrWatson
```
@quickactivate "project_name"
```
so that the command
```
include(projectdir("src", "HubbardFunctions.jl"))
import .HubbardFunctions as hf
```
includes the module ```HubbardFunctions```, independent of the current directory. This module stores all functions and structures. 

Once everything is properly initialized, we proceed by defining the model of interest. We destinguish between one-band and multi-band models.

## One-band
Each Hubbard model is linked to a structure, storing all its major properties. The first question we ask ourselves is whether we want to conserve the number of electrons by imposing a $U(1)$ symmetry, or if we want to add a chemical potential to deal with this. The first we achieve by using the ```OB_Sim()``` structure, the latter with ```OBC_Sim()```.

The most important properties of ```OB_Sim()``` are the hoppig ```t```, the Hubbard ```u```, the chemical potential ```µ```, and the filling defined by the ratio ```P```/```Q```.
```
t=[1.0, 0.1];
u=[8.0];
μ=0.0;

P=1;
Q=1;

s = 2.0;
bond_dim = 20;

model_OB = hf.OB_Sim(t, u, μ, P, Q, s, bond_dim; spin=false);
```
```t``` is a vector where the first element is the nearest neighbour hopping, the second the next-to-nearest neighbour hopping, and so on. Similarly, the first element of ```u``` is the on-site Coulomb repulsion and the second element the interaction between two neighbouring sites. 

The Schmidt cut ```s``` determines to which value the bond dimension is grown by the iDMRG2 algorithm, while ```bond_dim``` is the maximal value used for the initialization of the MPS.

Note that, in order to preserve injectivity, a unit cell of size ```Q``` is used if ```P```is even and of size ```2*Q``` if ```P``` is odd. Therefore, filling ratios that deviate from half filling ```P=Q=1``` tend to be more intensive.

Finally, the tag ```spin``` determines if spin up and down have to be treated independently. If ```spin=false```, an additional $SU(2)$ symmetry is imposed, reducing the local Hilbert space dimension to 3 and leading to a substantial speed up. However, no information about the spin of a state can be retrieved.

```MBC_Sim()``` works similarly. Now, we either provide a chemical potential or a filling.
```
model_OBC_1 = hf.OBC_Sim(t, u, P/Q, s, bond_dim; mu=false)
model_OBC_2 = hf.OBC_Sim(t, u, μ, s, bond_dim; mu=true)
```
If a filling is defined, the corresponding chemical potential is sought iteratively. Calculations without spin symmetry are not yet implemented.

## Multi-band
In analogy with the one-band model, multi-band models can be constructed using ```MB_Sim()``` or ```MBC_Sim```. For the one-band model, DrWatson is able to find a unique name for the model based on its parameters. This name is later used to retrieve earlier computed results. For multi-band models, the number of parameters is simply too large and we have to provide a unique name ourselves, like the name of the script for instance.
```
name_jl = last(splitpath(Base.source_path()))
name = first(split(name_jl,"."))
```
Then, we insert the parameters in the form of $B\times B$ matrices, where $B$ is the number of bands. For a 2-band model this looks as follows
```
t_onsite = [-0.784  2.588; 2.588 -0.784];
t_intersite = [0.0345 2.745; -0.036 0.0345];
t = cat(t_onsite,t_intersite, dims=2)
u = [9.86 5.7656; 5.7656 9.86];
J = [0.0 0.096; 0.096 0.0];
U13 = zeros(2,2);

P = 1;
Q = 1;

s=2.0;
bond_dim = 20;

model_MB = hf.MB_Sim(t, u, J, U13, P, Q, s, bond_dim; code = name, spin=false, force=false);
```
Where the one-band model used vectors for ```t``` and ```u```, the multi-band model concatenates matrices horizontally. In addition, the exchange $J$ and $U_{ijjj}$ parameters, with zeros on the diagonals as these are included in ```u```, are implemented as well. Since those parameters are usually rather small, ```U13``` is an optional argument.

When the parameters are changed but you want to keep the name of the model the same, you should put ```force=true``` to overwrite the previous results, obtained with the old parameters.

For a ```MBC_Sim``` structure, we would have
```
model_MBC = hf.MBC_Sim(t, u, J, s, bond_dim; code=name);
```
The chemical potential is included in the diagonal terms of ```t_onsite```. Iterative determination of the chemical potential for a certain filling is not yet supported.

## Ground state
The ground state of a model is computed (or loaded if it has been computed before) by the function ```produce_groundstate```. We can then extract the ground state energy as the expectation value of the Hamiltonian with respect to the ground state.
```
dictionary = hf.produce_groundstate(model);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = real(E0)./length(H);
println("Groundstate energy: $E")
```
Other properties, such as the bond dimension, the electron density and spin density (if it was a spin dependent calculation), can be calculated as well.
```
println("Bond dimension: $(hf.dim_state(ψ₀))")
println("Electron density: $(hf.density_state(ψ₀))")
println("Spin density: $(hf.density_spin(ψ₀))")
```

## Excited states
To compute quasiparticle excitations we have to choose the momentum, the number of excitations, and the symmetry sectors. 
```
resolution = 5;
momenta = range(0, π, resolution);
nums = 1;

exc = hf.produce_excitations(model, momenta, nums; charges=[0,0.0,0]);
Es = exc["Es"];
println("Excitation energies: ")
println(Es)
```
Excitations in the same sector as the ground state are found by defining ```charges``` as zeros. These charges refer to the difference with the ground state. Be aware that the meaning of the charges in this vector differ depending on the symmetries and thus on the type of model. ```OBC_Sim``` and ```MBC_Sim``` even have only two symmetries, and hence, two charges.

For example, a spin symmetric, electron conserving model has symmetry $\mathbb{Z}_2\times SU(2)\times U(1)$. Sectors differing by adding a particle or hole differ by ```charges = [1,1/2,+/-1]```. These single-particle excitations allow for the calculation of the band gap.
```
gap, k = hf.produce_bandgap(model)
println("Band Gap for s=$s: $gap eV at momentum $k")
```