import subprocess
import numpy as np
import numpy.random as npr
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
maxdim = 5

for i in range(repeat):

    dim = npr.randint(2, maxdim + 1)
    # max_time = 10.0 ** npr.uniform(-1, 1)
    max_time = npr.uniform(0.1, 1.0)
    trials = 20000
    # trials = 10
    dt = 0.0001

    # <dim> <max_time> <trials> <dt>
    job_list.append(f'{julia_com} MultiStateClassicalJump.jl {dim} {max_time} {trials} {dt}')


##### Setting end #####

def run_job(job):
    job_args = job.split()
    subprocess.run(args = job_args)

with ThreadPoolExecutor(max_workers = max_threads) as executor:
    [executor.submit(run_job, job) for job in job_list]
