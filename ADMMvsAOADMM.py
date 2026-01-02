import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt


from tensorly.decomposition import parafac
from tensorly.decomposition import ConstrainedCP
from admm_tensors import admm_tensor
from tensorly.cp_tensor import cp_to_tensor

## Generate a random tensor with a rank-r CPD
m = 20
n = 15
p = 8
r = 3
A = np.random.randn(m,r)
B = np.random.randn(n,r)
C = np.random.randn(p,r)
X = cp_to_tensor((None,[A,B,C]))

## Define ADMM for tensor parameters
error_measure = 'Frobenius'
parameters = {'rho':1 , 'maxiter':100}

T,A,B,C,objective,residual = admm_tensor(X,r,error_measure,parameters)
print("ADMM error=",objective[-1])

## Define AO-ADMM 
constrained_cp = ConstrainedCP(rank = r, n_iter_max=100, return_errors=True, tol_outer=-1, tol_inner=-1, init='random', cvg_criterion='abs_rec_error')
results = constrained_cp.fit_transform(X)
objective_aoadmm = constrained_cp.errors_
print("AO-ADMM error=",constrained_cp.errors_[-1])

## Plot comparison
iterations_admm = np.arange(len(objective))
iterations_aoadmm = np.arange(len(objective_aoadmm))

plt.figure(figsize=(10,5))

plt.semilogy(iterations_admm, objective, label='ADMM', linewidth=2)
plt.semilogy(iterations_aoadmm, objective_aoadmm, label='AO-ADMM', linewidth=2)

plt.title("ADMM vs AO-ADMM")
plt.xlabel("Iteration")
plt.ylabel("Objective value (log scale)")
plt.grid(True)
plt.legend()

plt.show()