import numpy as np
import matplotlib.pyplot as plt


from admm_tensors import admm_tensor
from ao_admm_tensors import ao_admm_tensor
from pytensorlab import PolyadicTensor

## Generate a random tensor with a rank-r CPD
m = 200
n = 150
p = 80
r = 10
A0 = np.random.randn(m,r)
B0 = np.random.randn(n,r)
C0 = np.random.randn(p,r)
U = PolyadicTensor([A0, B0, C0])
X = np.array(U)
## Initialization
Ainit = np.random.randn(m,r)
Binit = np.random.randn(n,r)
Cinit = np.random.randn(p,r)


## Define ADMM for tensor parameters
error_measure = 'Frobenius'
parameters = {'rho':1 , 'maxiter':100 , 'Ainit':Ainit, 'Binit':Binit, 'Cinit':Cinit}

T_admm,A_admm,B_admm,C_admm,objective_admm,residual_admm,time_history_admm = admm_tensor(X,r,error_measure,parameters)
print("ADMM error=",objective_admm[-1])

## Define AO-ADMM 
error_measure = 'Frobenius'
parameters = {'maxiter':100 , 'Ainit':Ainit, 'Binit':Binit, 'Cinit':Cinit}

T_aoadmm,A_aoadmm,B_aoadmm,C_aoadmm,objective_aoadmm,residual_aoadmm,time_history_aoadmm = ao_admm_tensor(X,r,error_measure,parameters)
print("AO-ADMM error=",objective_aoadmm[-1])

## Plot comparison
iterations_admm = np.arange(len(objective_admm))
iterations_aoadmm = np.arange(len(objective_aoadmm))

plt.figure(figsize=(10,5))

# ---- Left: Objective vs Iteration ----
plt.subplot(1, 2, 1)
plt.semilogy(iterations_admm, objective_admm, label='ADMM', linewidth=2)
plt.semilogy(iterations_aoadmm, objective_aoadmm, label='AO-ADMM', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Objective value")
plt.title("Objective vs Iteration")
plt.grid(True)
plt.legend()

# ---- Right: Objective vs Time ----
plt.subplot(1, 2, 2)
plt.semilogy(time_history_admm, objective_admm, label='ADMM', linewidth=2)
plt.semilogy(time_history_aoadmm, objective_aoadmm, label='AO-ADMM', linewidth=2)
plt.xlabel("Time (seconds)")
plt.ylabel("Objective value")
plt.title("Objective vs Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("progress_comparison.png", dpi=200)
plt.close()