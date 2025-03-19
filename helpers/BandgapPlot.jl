using EasyFit
using LaTeXStrings
using Printf
using Plots
using Plots.PlotMeasures
using DrWatson

@quickactivate "Hubbard_drW"

###############
# START INPUT #
###############

# Name of the material
material = "PT"

# [max bond dimension; band gap] of the different models
NN = [];
NNN = [];
FM = [];
TB = [16 40 80 144; 1.366 1.211696 1.097489 1.072567];

# Benchmarking data
DFT = 1.07533;
Exp = 2.1;

#############
# END INPUT #
#############

labels = ["NN hopping", "NN+NNN hopping", "Full model", "Tight binding"]
colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

Calc_vect = [NN, NNN, FM, TB]
Calculations = Dict()

for i in range(1,length(labels))
    Calculations[labels[i]] = Calc_vect[i]
end

println("-----------------------------------------")

hline([Exp], linestyle=:dash, linewidth=:1.5, color="black", label="Experiment", left_margin=[4mm 0mm])
hline!([DFT], linestyle=:dot, linewidth=:1.5, color="black", label="DFT")
for i in range(1,length(labels))
    if Calculations[labels[i]] != []
        X = copy(Calculations[labels[i]][1,:])
        Y = copy(Calculations[labels[i]][2,:])
        filter!(x->x≠0,X)
        filter!(x->x≠0,Y)
        fit = fitexp(X,Y)
        println(labels[i])
        println("Extrapolated band gap: $(fit.c)")
        println("Pearson correlation coefficient: $(fit.R)")
        println("-----------------------------------------")
        plot!(fit.x,fit.y, label="", color=:"gray", linewidth=:1.5)
        scatter!(X, Y, label=labels[i], mc=colors[i], ms=3, ma=10.0, m=:square)
    end
end
xlabel!("Max bond dimension")
ylabel!(L"$E_g$ (eV)")

savefig(joinpath(projectdir("plots","Band_gap"),material*"_Eg.pdf"))