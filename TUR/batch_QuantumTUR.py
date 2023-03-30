import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
argvs = sys.argv 
argc = len(argvs)

job_list = []

##### Setting start #####

julia_com = 'julia'

if argc == 1:
    max_threads = 8
else:
    max_threads = int(argvs[1])

def random_density_matrix(n):
    Are = np.random.rand(n, n)
    Aim = np.random.rand(n, n)
    A = Are + 1.0j * Aim
    B = A + A.conj().T
    
    evals, evecs = np.linalg.eig(B)
    if np.all(np.real(evals) >= 0.0):
        return B / np.trace(B)
    else:
        return random_density_matrix(n)

def c2s(val):
    val_str = f'{val}'
    return val_str.replace('(','').replace(')','').replace('j','im')

repeat = 1000

for i in range(repeat):
    Delta = np.random.uniform(low = 0.1,high = 3.0)
    Omega = np.random.uniform(low = 0.1,high = 3.0)
    kappa = np.random.uniform(low = 0.1,high = 3.0)
    max_time = np.random.uniform(0.1, 1.0)
    dt = 0.0001
    init_rho = random_density_matrix(2)
    R11 = c2s(init_rho[0,0])
    R21 = c2s(init_rho[1,0])
    R12 = c2s(init_rho[0,1])
    R22 = c2s(init_rho[1,1])
    trials = 100000
    # trials = 10

    # <Delta> <Omega> <kappa> <max_time> <dt> <R11> <R21> <R21> <R22> <trials>
    job_list.append(f'{julia_com} TwoStateAtomQuantumJump.jl {Delta} {Omega} {kappa} {max_time} {dt} {R11} {R21} {R12} {R22} {trials}')


##### Setting end #####

def run_job(job):
    job_args = job.split()
    subprocess.run(args = job_args)

with ThreadPoolExecutor(max_workers = max_threads) as executor:
    [executor.submit(run_job, job) for job in job_list]
