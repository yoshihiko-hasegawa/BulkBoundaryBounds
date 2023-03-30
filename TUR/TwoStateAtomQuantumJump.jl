using LinearAlgebra
using Random
using DelimitedFiles
using Statistics
using StatsBase
# using PyCall

include("DensitySSE.jl")
include("QDynamicalActivity.jl")

"""
Simulation of two-level atom driven by classical laser field
"""

function inner_prod(H, Ls, rho_init, t)
    op_num = size(Ls, 1)
    (dim, _) = size(H)
    tmp = zeros(ComplexF64, (dim, dim))
    tmp += H
    for c = 1:op_num
        tmp += -0.5im * Ls[c]' * Ls[c]
    end
    Heff = tmp
    # print(Heff)
    tmp2 = exp(-1.0im * Heff * t) * rho_init
    
    return abs(tr(tmp2))
end

function run(Delta, Omega, kappa, max_time, dt, init_rho0, trials)
    ev = [0, 1.0 + 0.0im]
    gv = [1.0 + 0.0im, 0]

    init_rho = init_rho0 / tr(init_rho0)

    H = Delta * ev * ev' + Omega / 2 * (ev * gv' + gv * ev')
    L = sqrt(kappa) * gv * ev'
    Ls = [L]

    rng = MersenneTwister()
    self = DensitySSE.init(H, Ls, dt, rng)

    jump_count_list = Int64[]
    
    for i = 1:trials
        jump_time_list, jump_id_list = DensitySSE.gen_jump_data(self, init_rho, max_time)
        jump_count = size(jump_id_list, 1)

        append!(jump_count_list, jump_count)
    end

    abs_inner_prod = inner_prod(H, Ls, init_rho, max_time)
    qactint = py"qactivity_integral_interp"(init_rho, H, L, max_time, max_time * 1.1)
    qact = py"qactivity_direct"(init_rho, H, L, max_time)

    str = "Delta_$(Delta)_Omega_$(Omega)_kappa_$(kappa)_Mtime_$(max_time)_dt_$(dt)_R11_$(init_rho[1,1])_R21_$(init_rho[2,1])_R12_$(init_rho[1,2])_R22_$(init_rho[2,2])_trials_$(trials)_mean_$(mean(jump_count_list))_var_$(var(jump_count_list))_qactint_$(qactint)_qact_$(qact)_absinner_$(abs_inner_prod)"
    str = replace(str, " "=>"")
    println(str)
end

Delta = 1.0
dt = 0.001
Omega = 1.0
kappa = 3.0 / 10
init_rho = [1.0 + 0.0im 0; 0 0]

if size(ARGS, 1) == 10
    Delta = parse(Float64, ARGS[1])
    Omega = parse(Float64, ARGS[2])
    kappa = parse(Float64, ARGS[3])
    max_time = parse(Float64, ARGS[4])
    dt = parse(Float64, ARGS[5])
    init_rho[1, 1] = parse(ComplexF64, ARGS[6])
    init_rho[2, 1] = parse(ComplexF64, ARGS[7])
    init_rho[1, 2] = parse(ComplexF64, ARGS[8])
    init_rho[2, 2] = parse(ComplexF64, ARGS[9])
    trials = parse(Int64, ARGS[10])
else
    println("<Delta> <Omega> <kappa> <max_time> <dt> <R11> <R21> <R12> <R22> <trials>")
    exit()
end

run(Delta, Omega, kappa, max_time, dt, init_rho, trials)
