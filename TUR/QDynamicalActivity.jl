using PyCall

"""
Calculation of quantum dynamical activity
"""

py"""
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy as sp
import scipy.integrate as spi
import scipy.linalg as spl
from scipy import interpolate
from sympy.physics.quantum import TensorProduct

def tensor_product(A, B):
    return TensorProduct(A, B)

def twosided_super_operator(H1, H2, Ls1_list, Ls2_list):
    dim, _ = H1.shape
    jump_op_num = len(Ls1_list)
    
    if len(Ls1_list) != len(Ls2_list):
        raise ValueError('The number of jump operators does not match')
    
    tmp = -1.0j*(tensor_product(np.eye(dim), H1) - tensor_product(H2.T, np.eye(dim)))
    
    for c in range(jump_op_num):
        L1 = Ls1_list[c]
        L2 = Ls2_list[c]
        tmp += tensor_product(L2.conj(), L1)
        tmp += -1/2 * tensor_product(np.eye(dim), (L1.conj().T @ L1))
        tmp += -1/2 * tensor_product((L2.conj().T @ L2).T, np.eye(dim))
        
    return tmp

def twosided_nsolve(super_op, rho_init, t):
    s, _ = rho_init.shape
    rho_init_vec = np.array(rho_init).T.flatten()
    tmp2 = (spl.expm(super_op * t) @ rho_init_vec).reshape(s, s).T
    return np.matrix(tmp2)

def QFI_direct(rho_init, H, L, t, tau, eps_ratio = 10**(-3)):
    eps = np.min([t * eps_ratio, 0.01])
    t1 = t
    t2 = t + eps

    H1 = t1/tau*H
    H2 = t2/tau*H
    L1 = np.sqrt(t1/tau)*L
    L2 = np.sqrt(t2/tau)*L
    sop = twosided_super_operator(H1, H2, [L1], [L2])
#     tmp_mat =  twosided_nsolve(sop, rho_init, t)
    tmp_mat =  twosided_nsolve(sop, rho_init, tau)

    return 8 / eps**2 * (1 - np.abs(np.trace(tmp_mat)))

def qactivity_integral(rho_init, H, L, tau, eps_ratio = 10**(-3), start_time = 10**(-5)):
    tmp = spi.quad(lambda t: np.sqrt(QFI_direct(rho_init, H, L, t, tau, eps_ratio = eps_ratio)), start_time, tau)[0]
    return 0.5 * tmp

def qactivity_integral_interp(rho_init, H, L, tau, interp_maxtime, divnum = 300, eps_ratio = 10**(-3), start_time = 10**(-5)):
    tslot = [_t for _t in np.linspace(start_time, interp_maxtime, num=divnum)]
    qfi_vals = [QFI_direct(rho_init, H, L, _t, tau, eps_ratio = eps_ratio) for _t in tslot]
    f = interpolate.interp1d(tslot, qfi_vals)
    tmp = spi.quad(lambda t: np.sqrt(f(t)), start_time, tau)[0]
    return 0.5 * tmp

def qactivity_direct(rho_init, H, L, tau, eps_ratio = 10**(-3)):
    return tau**2 * QFI_direct(rho_init, H, L, tau, tau, eps_ratio = eps_ratio)

def rho_ss(Delta, Omega, kappa):
    return np.mat([[1 / (4 * Delta ** 2 + 2 * Omega ** 2 + kappa ** 2) * (4 * Delta ** 2 + Omega ** 2 + kappa ** 2),1 / (4 * Delta ** 2 + 2 * Omega ** 2 + kappa ** 2) * (-2 * Delta * Omega + complex(0, 1) * kappa * Omega)],[1 / (4 * Delta ** 2 + 2 * Omega ** 2 + kappa ** 2) * (-2 * Delta * Omega + complex(0, -1) * kappa * Omega),1 / (4 * Delta ** 2 + 2 * Omega ** 2 + kappa ** 2) * Omega ** 2]])


"""    

