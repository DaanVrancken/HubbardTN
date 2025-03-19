using DrWatson
using Printf
@quickactivate "Hubbard"

include(projectdir("src", "HubbardFunctions.jl"))
import .HubbardFunctions as hf

# geometry = "tPA_PBE"
# geometry = "tPA_B3LYP_ODD_ev"
geometry = "tPA_CCSD"

geom_paper = Dict("tPA_PBE" => "tPA1", "tPA_B3LYP_ODD_ev" => "tPA2", "tPA_CCSD" => "tPA3")

input_dir = projectdir("helpers", "input", "params", geometry)
output_file = projectdir("helpers", "output", "params", geom_paper[geometry]*".txt")

include(joinpath(input_dir, "params.jl"))

# t = tmn + corr_H
t,_,_,_,_,_,_ = hf.extract_params(joinpath(input_dir, "params.jl"); range_u= 1, range_t=4, range_J=1, range_U13=1, r_1111 = 1, r_112 = 1)
t[1,8] = 0.0

function extract_and_write_latex(Wmn, filename, t; column=1)
    if column == 1
        error("not up to date")
        open(filename, "w") do io
            # Add the 2x8 matrix table
            println(io, "\\begin{table}[h]")
            println(io, "\\centering")
            println(io, "\\begin{tabular}{C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}}")
            println(io, "\\hline")
            println(io, "\\hline")
            println(io, "\\multicolumn{8}{c}{\$t\$}\\Tstrut \\\\")
            println(io, "\\multicolumn{2}{c}{[0,0,0]} & \\multicolumn{2}{c}{[1,0,0]}  & \\multicolumn{2}{c}{[2,0,0]} & \\multicolumn{2}{c}{[3,0,0]}\\Bstrut\\Tstrut \\\\")
            println(io, "\\cmidrule(lr){1-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}")

            for row in 1:2
                println(io, join([@sprintf "%.3f" t[row, col] for col in 1:8], " & ") * " \\\\")
            end
            
            println(io, "\\hline")
            println(io, "\\hline")
            println(io, "\\end{tabular}")
            println(io, "\\caption{Hopping parameters \$t^{\\alpha\\beta}_{ij}\$ included in the Hubbard model for $(geom_paper[geometry]). The elements (in eV) correspond to the parameter between band \$\\alpha\$ on site \$\\mathbf{R}_i=[0,0,0]\$ (rows) and band \$\\beta\$ on different sites \$\\mathbf{R}_j=[j,0,0]\$ (columns). \\label{tab:hop$(geom_paper[geometry])}}")
            println(io, "\\end{table}\n")

            println(io, "\\begin{longtable}{ccccr}")
            println(io, "\\hline")
            println(io, "\\hline")
            println(io, "\$(2i+\\alpha)\$ & \$(2j+\\beta)\$ & \$(2k+\\gamma)\$ & \$(2l+\\delta)\$ & \$U_{ijkl}^{\\alpha\\beta\\gamma\\delta}\$ \\\\")
            println(io, "\\hline")
            println(io, "\\hline")
            
            for i in 3:7, j in 3:7, k in 3:7, l in 3:7, a in 1:2, b in 1:2, c in 1:2, d in 1:2
                if i == 3 || j == 3 || k == 3 || l == 3  # At least one index is 3
                    if (maximum([i*2+a, j*2+b, k*2+c, l*2+d]) - minimum([i*2+a, j*2+b, k*2+c, l*2+d])) <= 4  # range of 4
                        value = Wmn[i, j, k, l, a, b, c, d]
                        if abs(value) >= 0.01  # Exclude small values
                            println(io, @sprintf "%d & %d & %d & %d & %.3f \\\\" 2*mod(i,3)+a 2*mod(j,3)+b 2*mod(k,3)+c 2*mod(l,3)+d value)
                        end
                    end
                end
            end

            println(io, "\\hline")
            println(io, "\\hline")
            println(io, "\\caption{Interaction parameters (in eV) included in the Hubbard model for $(geom_paper[geometry]). \\label{tab:values$(geom_paper[geometry])}}")
            println(io, "\\end{longtable}")
        end
    else
        open(filename, "w") do io
            # Add the 2x8 matrix table
            println(io, "\\begin{table}[h]")
            println(io, "\\caption{Hopping parameters \$t^{\\alpha\\beta}_{ij}\$ included in the Hubbard model for $(geom_paper[geometry]). The elements (in eV) correspond to the parameter between band \$\\alpha\$ on site \$\\mathbf{R}_i=[0,0,0]\$ (rows) and band \$\\beta\$ on different sites \$\\mathbf{R}_j=[j,0,0]\$ (columns). \\label{tab:hop$(geom_paper[geometry])}}")
            println(io, "\\begin{ruledtabular}")
            println(io, "\\begin{tabular}{C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}}")
            println(io, "\\multicolumn{8}{c}{\$t\$}\\Tstrut \\\\")
            println(io, "\\multicolumn{2}{c}{[0,0,0]} & \\multicolumn{2}{c}{[1,0,0]}  & \\multicolumn{2}{c}{[2,0,0]} & \\multicolumn{2}{c}{[3,0,0]}\\Bstrut\\Tstrut \\\\")
            println(io, "\\cmidrule(lr){1-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}")

            for row in 1:2
                println(io, join([@sprintf "%.3f" t[row, col] for col in 1:8], " & ") * " \\\\")
            end
            
            println(io, "\\end{tabular}")
            println(io, "\\end{ruledtabular}")
            println(io, "\\end{table}\n")

            println(io, "\\begin{longtable}{ccccrccccr}")
            println(io, "\\caption{Interaction parameters (in eV) included in the Hubbard model for $(geom_paper[geometry]). \\label{tab:values$(geom_paper[geometry])}} \\\\")
            println(io, "\\hline")
            println(io, "\\hline")
            println(io, "\$(2i+\\alpha)\$ & \$(2j+\\beta)\$ & \$(2k+\\gamma)\$ & \$(2l+\\delta)\$& \$U_{ijkl}^{\\alpha\\beta\\gamma\\delta}\$ & \$(2i+\\alpha)\$ & \$(2j+\\beta)\$ & \$(2k+\\gamma)\$ & \$(2l+\\delta)\$ & \$U_{ijkl}^{\\alpha\\beta\\gamma\\delta}\$ \\\\")
            println(io, "\\cmidrule(lr){1-5} \\cmidrule(lr){6-10}")
            
            counter = 0
            i1 = 0.0
            j1 = 0.0
            k1 = 0.0
            l1 = 0.0
            value1 = 0.0
            for i in 3:7, j in 3:7, k in 3:7, l in 3:7, a in 1:2, b in 1:2, c in 1:2, d in 1:2
                if i == 3 || j == 3 || k == 3 || l == 3  # At least one index is 3
                    if (maximum([i*2+a, j*2+b, k*2+c, l*2+d]) - minimum([i*2+a, j*2+b, k*2+c, l*2+d])) <= 4  # range of 4
                        value = Wmn[i, j, k, l, a, b, c, d]
                        if abs(value) >= 0.01  # Exclude small values
                            if iseven(counter)
                                i1 = 2*mod(i,3)+a
                                j1 = 2*mod(j,3)+b
                                k1 = 2*mod(k,3)+c
                                l1 = 2*mod(l,3)+d
                                value1 = value
                                counter += 1
                            else
                                println(io, @sprintf "%d & %d & %d & %d & %.3f & %d & %d & %d & %d & %.3f \\\\" i1 j1 k1 l1 value1 2*mod(i,3)+a 2*mod(j,3)+b 2*mod(k,3)+c 2*mod(l,3)+d value)
                                counter += 1
                            end
                        end
                    end
                end
            end
            println(io, "\\hline")
            println(io, "\\hline")
            println(io, "\\end{longtable}")
        end
    end
end


extract_and_write_latex(Wmn, output_file, t; column=2)