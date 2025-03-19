using DrWatson
@quickactivate "HubbardMPS"

using Printf
using LinearAlgebra

include(projectdir("src", "HubbardFunctions.jl"))
import .HubbardFunctions as hf

path = "params_all.jl"
output_file = "params_r3.jl"

t,U,J,U13,U13_IS,U112_o,U1111_o = hf.extract_params(path; range_u=3, range_t=3, range_J=2, range_U13=2, r_112=2, r_1111=2)

# reored basis
function permute_submatrices(mat, new_order)
    B = length(new_order)
    P = zeros(Int, B, B)
    for i in 1:B
        P[i, new_order[i]] = 1
    end

    # Check that the number of columns is divisible by B
    cols = size(mat, 2)
    n = cols ÷ B
    @assert cols == B * n "Matrix columns must be divisible by B"

    # Reshape to a 3D array (B × B × n)
    reshaped = reshape(mat, B, B, n)

    for i in 1:n
        reshaped[:,:,i] = P * reshaped[:,:,i] * P'
    end

    # Reshape back to B × (B * n) matrix
    return reshape(reshaped, B, B * n)
end

new_order = [3, 1, 6, 2, 4, 5]
t = permute_submatrices(t, new_order)
U = permute_submatrices(U, new_order)
J = permute_submatrices(J, new_order)
U13 = permute_submatrices(U13, new_order)
for i in 1:4
    U13_IS[:,:,i] = permute_submatrices(U13_IS[:,:,i], new_order)
end

U112 = copy(U112_o)
U1111 = copy(U1111_o)
new_order = [3, 1, 6, 2, 4, 5, 9, 7, 12, 8, 10, 11]
for key in keys(U112)
    U112[key] = U112_o[(new_order[key[1]],new_order[key[2]],new_order[key[3]],new_order[key[4]])]
end
for key in keys(U1111)
    U1111[key] = U1111_o[(new_order[key[1]],new_order[key[2]],new_order[key[3]],new_order[key[4]])]
end

open(output_file, "w") do julia_file
    # Write a header for the Julia script
    write(julia_file, "# Julia arrays generated from .npy files\n\n")
    write(julia_file, @sprintf("%s = %s\n\n", "t", repr(t)))
    write(julia_file, @sprintf("%s = %s\n\n", "U", repr(U)))
    write(julia_file, @sprintf("%s = %s\n\n", "J", repr(J)))
    write(julia_file, @sprintf("%s = %s\n\n", "U13", repr(U13)))
    write(julia_file, @sprintf("%s = %s\n\n", "U13_IS", repr(U13_IS)))
    write(julia_file, @sprintf("%s = %s\n\n", "U112", repr(U112)))
    write(julia_file, @sprintf("%s = %s\n\n", "U1111", repr(U1111)))
end

println("Julia script generated at $output_file")