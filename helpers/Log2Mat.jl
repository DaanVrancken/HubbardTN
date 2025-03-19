using LinearAlgebra

# Define useful functions

function extract_real_value(complex_str)
    clean_str = replace(complex_str, r"[\(\)]" => "")
    complex_value = parse(Complex{Float64}, clean_str)

    return real(complex_value)
end

function int_matrix(filename, patterns, matrix_type)
    file_content = open(filename, "r") do file
        read(file, String)
    end

    pattern = patterns[matrix_type]
    matches = eachmatch(pattern, file_content)

    W_wan_dict = Dict{Tuple{Int, Int}, Float64}()
    for m in matches
        i = parse(Int, m.captures[1])
        j = parse(Int, m.captures[2])
        W_wan_dict[(i, j)] = extract_real_value(m.captures[3])
    end

    Bands = maximum(k[1] for k in keys(W_wan_dict))
    M = zeros(Float64, Bands, Bands)

    for ((i, j), value) in W_wan_dict
        M[i, j] = value
    end

    if matrix_type != "U"
        M[diagind(M)] .= 0
    end

    return M
end

function hop_matrix(filename)
    file_content = open(filename, "r") do file
        read(file, String)
    end

    tij_pattern = r"t(\d)(\d)\[(\d) 0 0\]=\(([^)]+)\)"
    matches = eachmatch(tij_pattern, file_content)

    tij_dict = Dict{Tuple{Int, Int, Int}, Complex{Float64}}()
    for m in matches
        i = parse(Int, m.captures[1])
        j = parse(Int, m.captures[2])
        site = parse(Int, m.captures[3])
        tij_dict[(i, j, site)] = extract_real_value(m.captures[4])
    end

    Bands = maximum(k[1] for k in keys(tij_dict))
    range = maximum(k[3] for k in keys(tij_dict))
    t = zeros(Float64, Bands, Bands*(range+1))

    for ((i, j, site), value) in tij_dict
        t[i, j+site*Bands] = value
    end

    return t
end

function extract_name(filename)
    file_content = read(filename, String)

    pattern = r"results\/(.*?).*_PBE\.chk"
    match_result = findfirst(pattern, file_content)
    
    if match_result !== nothing
        string = file_content[match_result]
        name = string[9:(findfirst("_",string)[1]-1)]
        simulation = match(r"[^/\\]+(?=[/\\][^/\\]+$)", string)
        return name, simulation !== nothing ? simulation.match : "nameless"
    else
        return "nameless", "nameless"
    end
end

# Define text patterns
patterns = Dict("U"=>r"W_wan(\d)\1(\d)\2=\(([^)]+)\)",
            "J"=>r"W_wan(\d)(\d)\1\2=\(([^)]+)\)",
            "J2"=>r"W_wan(\d)(\d)\2\1=\(([^)]+)\)",
            "U13"=>r"W_wan(\d)(\d)\2\2=\(([^)]+)\)",
            "U13b"=>r"W_wan(\d)(\d)\1\1=\(([^)]+)\)",
            "U13c"=>r"W_wan(\d)\1(\d)\1=\(([^)]+)\)",
            "U13d"=>r"W_wan(\d)\1\1(\d)=\(([^)]+)\)",)

# Determine matrices
filename = joinpath("Input","log.txt")
U_matrix = int_matrix(filename, patterns, "U")
J_matrix = int_matrix(filename, patterns, "J")
U13_matrix = int_matrix(filename, patterns, "U13")'

# Check for discrepancies
J2 = int_matrix(filename, patterns, "J2")
prod(isapprox.(J_matrix,J2;rtol=0.001)) == 0 && @warn "At least one element of J1 and J2 defer by more than 0.1%."
for perm in ["U13b","U13c","U13d"]
    u_perm = int_matrix(filename, patterns, perm)
    if prod(isapprox.(U13_matrix,u_perm;rtol=0.001)) == 0 
        @warn "At least one element of U13(a) and "*perm*" defer by more than 0.1%."
    end
end

# Print the results
system, simulation = extract_name(filename)

output_dir = joinpath("Output","Model_"*system*".txt")
open(output_dir,"a") do io
    println(io,"--------------------")
    println(io,"Simulation: ",simulation)
    println(io,"Hopping matrix")
    println(io,hop_matrix(filename))
    println(io,"Interaction matrices")
    println(io,"U:")
    println(io,U_matrix)
    println(io,"J:")
    println(io,J_matrix)
    println(io,"U13:")
    println(io,U13_matrix)
end