using LinearAlgebra

function twosided_super_operator(H1, H2, Ls1_list, Ls2_list)
    dim = size(H1, 1)
    jump_op_num = size(Ls1_list, 1)
    Id = Matrix{ComplexF64}(I, dim, dim)
    
    tmp = -1.0im * (kron(Id, H1) - kron(transpose(H2), Id))
    
    for c in 1:jump_op_num
        L1 = Ls1_list[c]
        L2 = Ls2_list[c]
        tmp += kron(conj(L2), L1)
        tmp += -0.5 * kron(Id, L1' * L1)
        tmp += -0.5 * kron(transpose(L2' * L2), Id)
    end
    
    return tmp
end

function twosided_nsolve(super_op, rho_init, t)
    s = size(rho_init, 1)
    rho_init_vec = vec(rho_init)
    tmp = (exp(super_op * t) * rho_init_vec)
    tmp = reshape(tmp, (s, s))
    return tmp
end
