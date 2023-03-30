module DensitySSE

using LinearAlgebra
using Random
using StatsBase

"""
Simulation of stochastic Schrodinger equation
"""

mutable struct Self
    dim::Int64
    jump_num::Int64
    H::Array{ComplexF64, 2}
    Ls::Array{Array{ComplexF64, 2}, 1}
    dt::Float64
    rng::AbstractRNG
end

function init(
        H::Array{ComplexF64, 2}, 
        Ls::Array{Array{ComplexF64, 2}, 1},
        dt::Float64,
        rng::AbstractRNG
    )
    dim = size(H, 1)
    jump_num = size(Ls, 1)
    
    return Self(dim, jump_num, H, Ls, dt, rng)
end

function next_val(
        self::Self, 
        rho::Array{ComplexF64, 2}
    )
    dt = self.dt
    dim = self.dim
    jump_num = self.jump_num
    rng = self.rng
    Ls = self.Ls
    H = self.H
    
    delta_p_array = zeros(Float64, jump_num)
    for i = 1:jump_num
        L = Ls[i]
        delta_p_array[i] = real(tr(L * rho * L')) * dt
    end

    delta_p = sum(delta_p_array)
    rv = rand(rng)
    if rv < delta_p # Jump occurs
        i = sample(collect(1:jump_num), Weights(delta_p_array))
        L = Ls[i]
        next_rho = L * rho * L' / (real(tr(L * rho * L')))
        return next_rho, i
    else # Jump does not occur
        phi1 = rho
        phi1 += -1.0im * (H * rho - rho * H) * dt
        for i = 1:jump_num
            L = Ls[i]
            phi1 += -0.5 * (L' * L * rho + rho * L' * L) * dt
            phi1 += rho * tr(L * rho * L') * dt
        end
        
        return phi1 / tr(phi1), 0
    end    
end

function gen_traj(
        self::Self,
        init_rho::Array{ComplexF64, 2}, 
        max_time::Float64
    )
    dt = self.dt
    dim = self.dim    
    rsize = convert(Int64, round(max_time / dt))

    traj = zeros(ComplexF64, rsize, dim^2 + 1)
    t = 0.0
    rho = copy(init_rho)
    for i in 1:rsize
        rho, jump_id = next_val(self, rho)
        t += dt
        traj[i, 1] = t
        traj[i, 2:end] = reshape(rho, (dim^2, 1))
    end

    return traj
end

function gen_jump_data(
        self::Self,
        init_rho::Array{ComplexF64, 2}, 
        max_time::Float64
    )
    dt = self.dt
    dim = self.dim
    jump_id_list = Int64[]
    jump_time_list = Float64[]
    
    rsize = convert(Int64, round(max_time / dt))
#     traj = zeros(ComplexF64, rsize, dim^2 + 1)
    t = 0.0
    rho = copy(init_rho)
    for i in 1:rsize
        rho, jump_id = next_val(self, rho)
        if jump_id > 0
            append!(jump_id_list, jump_id)
            append!(jump_time_list, t)
        end
        t += dt
#         traj[i, 1] = t
#         traj[i, 2:end] = reshape(rho, (dim^2, 1))
    end

#     return traj
    return jump_time_list, jump_id_list
end

end

