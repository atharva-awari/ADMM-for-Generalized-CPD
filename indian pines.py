from scipy.io import loadmat
import numpy as np
import pytensorlab as ptl 
import matplotlib.pyplot as plt
from admm_tensors import admm_tensor
from ao_admm_tensors import ao_admm_tensor
from pytensorlab import PolyadicTensor
from pytensorlab import CPDAlsOptions
from pytensorlab import CPDNlsOptions
from pytensorlab import CPDMinfOptions
import time

data = loadmat("Indian_pines_corrected.mat")
X = data["indian_pines_corrected"]
m,n,p = X.shape
r=10
print(X.shape)  # (145, 145, 200)

normX = ptl.frob(X)

## Initialization
Ainit = np.random.randn(m,r)
Binit = np.random.randn(n,r)
Cinit = np.random.randn(p,r)
Uinit = PolyadicTensor([Ainit,Binit,Cinit])

## Define ADMM for tensor parameters
tic = time.perf_counter()
error_measure = 'Frobenius'
parameters = {'rho':1 , 'maxiter':np.inf, 'maxtime': 10, 'Ainit':Ainit, 'Binit':Binit, 'Cinit':Cinit}
T_admm,A_admm,B_admm,C_admm,objective_admm,residual_admm,time_history_admm = admm_tensor(X,r,error_measure,parameters)
time_admm = time.perf_counter() - tic
# Downsample every 30 iterations
#step = 15
#obj_admm_ds = objective_admm[step-1::step]   # 30, 60, 90, ...
# Prepend initialization
#objective_admm_ds = np.insert(obj_admm_ds, 0, objective_admm[0])

## Define AO-ADMM 
tic = time.perf_counter()
error_measure = 'Frobenius'
parameters = {'maxiter':np.inf, 'inner_iter':5, 'maxtime':10, 'Ainit':Ainit, 'Binit':Binit, 'Cinit':Cinit}
T_aoadmm,A_aoadmm,B_aoadmm,C_aoadmm,objective_aoadmm,residual_aoadmm,time_history_aoadmm = ao_admm_tensor(X,r,error_measure,parameters)
time_aoadmm = time.perf_counter() - tic

## Define CPD-ALS parameters
tic = time.perf_counter()
als_options = CPDAlsOptions(max_iterations=100, tol_relfval=1e-16, tol_relstep=1e-16, tol_absfval=0.0, display=0) 
Uopt_als,logger_als = ptl.cpd_als(X,Uinit,options=als_options)
time_als = time.perf_counter() - tic
fvalue_als = logger_als.fval
niterations_als = logger_als.niterations
relerr_als = np.sqrt(2 * np.array(fvalue_als)) / normX

## Define CPD-NLS parameters
tic = time.perf_counter()
nls_options = CPDNlsOptions(max_iterations=1,large_scale=False, tol_relfval=1e-16, tol_relstep=1e-16, tol_absfval=0.0, max_internal_iterations=10, tol_internal_residual=1e-16, display=0)
Uopt_nls, logger_nls = ptl.cpd_nls(X,Uinit,options=nls_options)
time_nls = time.perf_counter() - tic
fvalue_nls = logger_nls.fval
niterations_nls = logger_nls.niterations
relerr_nls = np.sqrt(2 * np.array(fvalue_nls)) / normX

## Define CPD-NLS SciPy parameters
tic = time.perf_counter()
Uopt_nlsscipy, logger_nlsscipy = ptl.cpd_nls_scipy(X, Uinit, options=nls_options, method='trust-ncg') 
time_nlsscipy = time.perf_counter() - tic
fvalue_nlsscipy = logger_nlsscipy.fval
niterations_nlsscipy = logger_nlsscipy.niterations
relerr_nlsscipy = np.sqrt(2 * np.array(fvalue_nlsscipy)) / normX

## Define CPD-Minf parameters
tic = time.perf_counter()
minf_options = CPDMinfOptions(method='l-bfgs-b', memory_limit=30, max_iterations=100, tol_relfval=1e-16, tol_relstep=1e-16, tol_absfval=0.0,display=0)
Uopt_minf,logger_minf = ptl.cpd_minf(X,Uinit, options=minf_options)
time_minf = time.perf_counter() - tic
fvalue_minf = logger_minf.fval
niterations_minf = logger_minf.niterations
relerr_minf = np.sqrt(2 * np.array(fvalue_minf)) / normX

# =====================================================
# Display runtimes
# =====================================================
print("\n========= Runtime Summary =========")
print(f"ADMM             : {time_admm:.2f} seconds")
print(f"AO-ADMM          : {time_aoadmm:.2f} seconds")
print(f"CPD-ALS          : {time_als:.2f} seconds")
print(f"CPD-NLS          : {time_nls:.2f} seconds")
print(f"CPD-NLS SciPy    : {time_nlsscipy:.2f} seconds")
print(f"CPD-Minf         : {time_minf:.2f} seconds")
print("===================================\n")

## Save errors in a text file
np.savetxt("cpdalg_admm.txt", objective_admm)
np.savetxt("cpdalg_aoadmm.txt", objective_aoadmm)
np.savetxt("cpdalg_als.txt", relerr_als)
np.savetxt("cpdalg_nls.txt", relerr_nls)
np.savetxt("cpdalg_nlsscipy.txt", relerr_nlsscipy)
np.savetxt("cpdalg_minf.txt", relerr_minf)


## Plot comparison
iterations_admm = np.arange(len(objective_admm))
iterations_aoadmm = np.arange(len(objective_aoadmm))
iterations_als = np.arange(len(relerr_als))
iterations_nls = np.arange(len(relerr_nls))
iterations_nlsscipy = np.arange(len(relerr_nlsscipy))
iterations_minf = np.arange(len(relerr_minf))


plt.figure(figsize=(10,5))

#  Objective vs Iteration 
plt.semilogy(iterations_admm, objective_admm, label='ADMM', linewidth=2)
plt.semilogy(iterations_aoadmm, objective_aoadmm, label='AO-ADMM', linewidth=5)
plt.semilogy(iterations_als, relerr_als, label='CPD-ALS', linewidth=2)
plt.semilogy(iterations_nls, relerr_nls, label='CPD-NLS', linewidth=2)
plt.semilogy(iterations_nlsscipy, relerr_nlsscipy, label='CPD-NLS SciPy', linewidth=2)
plt.semilogy(iterations_minf, relerr_minf, label='CPD-Minf', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Relative error")
plt.title("Relative error vs Iteration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ADMMvsCPDalgorithms.png", dpi=200)
plt.close()