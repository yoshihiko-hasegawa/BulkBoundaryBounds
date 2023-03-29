using PyCall

py"""
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy as sp
import scipy.integrate as spi
import scipy.linalg as spl

def test(x):
    return x

def steady_state_distribution(transition_rate_matrix): 
    TRM = np.matrix(transition_rate_matrix)
    evs, evecs = npl.eig(TRM)
    zero_index = np.argmin(np.abs(evs))
    
    tmp = (evecs[:, zero_index]).A1 / np.sum((evecs[:, zero_index]).A1)
    return tmp.real

class DynamicalActivityCalc:

    def solve_equation(self, W, P_init):
        return lambda t: spl.expm(W * t) @ P_init

    def activity(self, W, P_init, max_time):
        Pt = self.solve_equation(W, P_init)
        return spi.quad(lambda t: self.instant_activity(W, Pt(t)), 0, max_time)[0]

    def instant_activity(self, W, P_at_t):
        N = len(P_at_t)
        tmp = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    tmp += P_at_t[i] * W[j, i]
        return tmp

    def activity_integral(self, W, P_init, tau):
        tmp = spi.quad(lambda t: np.sqrt(1 / t**2 * self.activity(W, P_init, t)), 0, tau)[0]
        return 0.5 * tmp
"""    

