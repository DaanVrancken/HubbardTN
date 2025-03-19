using LinearAlgebra
using Printf

# Function to read the POSCAR file and extract data
function read_poscar(filename::String)
    open(filename, "r") do file
        lines = readlines(file)

        title = lines[1]
        scaling_factor = parse(Float64, lines[2])
        lattice_vectors = [parse.(Float64, split(lines[i])) for i in 3:5]
        lattice_matrix = hcat(lattice_vectors...)

        atomic_species = split(lines[6])
        num_atoms = parse.(Int, split(lines[7]))

        coordinate_type = strip(lines[8])  # Check if it's 'Direct' or 'Cartesian'

        atomic_positions = [parse.(Float64, split(lines[i])) for i in 9:8 + sum(num_atoms)]

        return title, scaling_factor, lattice_matrix, atomic_species, num_atoms, coordinate_type, atomic_positions
    end
end

# Function to convert direct coordinates to cartesian coordinates
function direct_to_cartesian(lattice_matrix::Matrix{Float64}, direct_coords::Vector{Vector{Float64}})
    cartesian_coords = [lattice_matrix * vec for vec in direct_coords]
    return cartesian_coords
end

# Function to write the new POSCAR file with Cartesian coordinates
function write_poscar(filename::String, title, scaling_factor, lattice_matrix, atomic_species, num_atoms, cartesian_coords)
    open(filename, "w") do file
        write(file, title * "\n")
        write(file, string(scaling_factor) * "\n")
        for i in 1:3
            write(file, @sprintf("%15.8f %15.8f %15.8f\n", lattice_matrix[1, i], lattice_matrix[2, i], lattice_matrix[3, i]))
        end

        write(file, join(atomic_species, " ") * "\n")
        write(file, join(num_atoms, " ") * "\n")
        write(file, "Cartesian\n")

        for coords in cartesian_coords
            write(file, @sprintf("%15.8f %15.8f %15.8f\n", coords...))
        end
    end
end

# Main script to read, convert, and write the new POSCAR
input_filename = joinpath("Input","POSCAR")
output_filename = joinpath("Output","POSCAR")

title, scaling_factor, lattice_matrix, atomic_species, num_atoms, coordinate_type, atomic_positions = read_poscar(input_filename)

if lowercase(coordinate_type) != "direct"
    error("Input file must have direct coordinates")
end

cartesian_coords = direct_to_cartesian(lattice_matrix, atomic_positions)
write_poscar(output_filename, title, scaling_factor, lattice_matrix, atomic_species, num_atoms, cartesian_coords)
