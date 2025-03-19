# load pymatgen
# for other module to work, use: 
# ml "module", which python
# ENV["PYTHON"] = "/path/to/python"
# Pkg.build("PyCall")

using DrWatson
@quickactivate "Hubbard"

using PyCall
using Printf
np = pyimport("numpy")

function npy_to_julia_script(input_dir::String, output_file::String)
    """
    Convert all .npy files in a directory to Julia arrays in a .jl script.

    Parameters:
    - input_dir: String, directory containing .npy files
    - output_file: String, path to the output .jl file
    """
    open(output_file, "w") do julia_file
        # Write a header for the Julia script
        write(julia_file, "# Julia arrays generated from .npy files\n\n")

        # Loop through all .npy files in the directory
        for file in readdir(input_dir)
            if endswith(file, ".npy")
                # Load the NumPy array
                file_path = joinpath(input_dir, file)
                array = np.load(file_path)

                # Convert the file name to a valid Julia variable name
                variable_name = replace(basename(file), ".npy" => "")
                variable_name = replace(variable_name, r"[\s\-]" => "_")

                # Convert the NumPy array to a Julia array
                julia_array = real(convert(Array{ComplexF64}, array))

                if variable_name=="Wmn" && length(size(julia_array))==9
                    julia_array = julia_array[1,:,:,:,:,:,:,:,:] 
                end

                # Write the Julia array to the file
                write(julia_file, @sprintf("%s = %s\n\n", variable_name, repr(julia_array)))
            end
        end
    end

    println("Julia script generated at $output_file")
end

# Usage
input_directory = "."  # Replace with the path to your .npy files
output_julia_script = "params_all.jl"          # Replace with your desired output .jl file name
npy_to_julia_script(input_directory, output_julia_script)
