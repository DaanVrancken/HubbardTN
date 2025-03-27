# HubbardTN

Contains code for constructing and solving general one-dimensional Hubbard models using matrix product states. The framework is built upon [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl) and [TensorKit.jl](https://github.com/jutho/TensorKit.jl). The functionalities include constructing a Hubbard Hamiltonian with arbitrary interactions represented by the tensor U<sub>ijkl</sub>, as well as enabling hopping and interactions beyond nearest neighbors. Additionally, the framework supports U(1) symmetry for particle conservation and SU(2) spin symmetry. Check out the examples for concrete use-cases. More information can be found in the docs.

To (locally) reproduce this project, do the following:

1. Download this code base.
   ```
   git clone https://github.com/DaanVrancken/HubbardTN
   ```
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "Hubbard"
```
which auto-activate the project and enable local path handling from DrWatson.
