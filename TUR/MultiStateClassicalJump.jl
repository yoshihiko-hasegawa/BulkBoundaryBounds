using LinearAlgebra
using Random
using DelimitedFiles
using Statistics
using StatsBase
using PyCall

include("DensitySSE.jl")
include("MarkovChainGenerator.jl")
include("DynamicalActivity.jl")

"""
Simulation of classical Markov process
"""

function classical_markov_to_lindblad(transition_rate_matrix, EPS = 1.0E-6)
#     EPS = 1.0E-6
    W = copy(transition_rate_matrix)
    dim = size(W, 1)
    Id = diagm([1 for _ in 1:dim])
    ket = i -> Id[:, i]
    bra = i -> ket(i)'
    op_list = Array{ComplexF64, 2}[]
    for i in 1:dim
        for j in 1:dim
            if i != j && W[j, i] > EPS
                tmp = sqrt(W[j, i]) * ket(j) * bra(i)
                push!(op_list, tmp)
            end
        end
    end
    
    return op_list
end

function apply_projector(proj::Array{Float64}, jump_id_list::Array{Int64})
    tmp = 0
    jump_num = size(jump_id_list, 1)
#     print(jump_num)
    for i = 1:jump_num
        tmp += proj[jump_id_list[i]]
    end
    return tmp
end

function run(
    dim::Int64, 
    max_time::Float64,
    trials::Int64,
    dt::Float64
)
    rng = MersenneTwister()

    gen = py"MarkovChainGenerator"()
    G = gen.random_connected_graph(dim)
    W = gen.transition_rate_matrix(G)

    H_empty = zeros(ComplexF64, (dim, dim))
    Ls = classical_markov_to_lindblad(W)
    jump_op_num = size(Ls, 1)
    projector = [rand(rng) for _ in 1:jump_op_num]

    # init_rho = py"random_hermite"(dim)
    P_init = [rand(rng) for _ in 1:dim]
    P_init = P_init / sum(P_init)
    init_rho = Diagonal{ComplexF64}(P_init) + zeros(ComplexF64, (dim, dim))
    
    self = DensitySSE.init(H_empty, Ls, dt, rng)

    counting_val_list = Float64[]
    for i = 1:trials
        jump_time_list, jump_id_list = DensitySSE.gen_jump_data(self, init_rho, max_time)
        cval = apply_projector(projector, jump_id_list)
        append!(counting_val_list, cval)
    end

    da = py"DynamicalActivityCalc"()
    aint = da.activity_integral(W, P_init, max_time)
    act = da.activity(W, P_init, max_time)

    str = "dim_$(dim)_maxtime_$(max_time)_trials_$(trials)_dt_$(dt)_mean_$(mean(counting_val_list))_var_$(var(counting_val_list))_actint_$(aint)_act_$(act)"
    str = replace(str, " "=>"")
    println(str)
end

dim = 3
max_time = 10.0
trials = 10000
dt = 0.0001

if size(ARGS, 1) == 4
    dim = parse(Int64, ARGS[1])
    max_time = parse(Float64, ARGS[2])
    trials = parse(Int64, ARGS[3])
    dt = parse(Float64, ARGS[4])
else
    println("<dim> <max_time> <trials> <dt>")
    exit()
end

run(dim, max_time, trials, dt)
