import numpy as np
import matplotlib.pyplot as plt
import pytensorlab as ptl
import time 

from pytensorlab import cpd, cpd_als
from pytensorlab import PolyadicTensor
from update_T import update_T
from compute_objective_residual import compute_objective_residual


def admm_tensor(X,r,error_measure,parameters):
    m,n,p = X.shape
    rho = parameters.get('rho', 1.0)
    maxiter = parameters.get('maxiter', 1000)
    maxtime = parameters.get('maxtime', 10)
    A   = parameters.get('Ainit', np.random.randn(m,r))
    B   = parameters.get('Binit', np.random.randn(n,r))
    C   = parameters.get('Cinit', np.random.randn(p,r))
    T = X
    Lambda= np.zeros((m,n,p))
   
    ## For KL divergence T has to be initialized with positive entries where X is positive
    if error_measure == 'KL divergence':
       f = X > 0
       T[f] = np.maximum(np.finfo(float).eps, T[f])


    k=0
    start_time = time.time()
    U = PolyadicTensor([A, B, C])
    Xhat = np.array(U)
    objective = []
    residual  = []
    time_history = []
    obj,res= compute_objective_residual(X,Xhat,T,error_measure)
    objective.append(obj)
    residual.append(res)
    time_history.append(0.0)
    while k<maxiter and (time.time() - start_time < maxtime):
       
        Ures, info = cpd(T + Lambda,nterm=r, optimization=cpd_als,compression=False,print_steps=False)

        A, B, C = Ures.factors
        Xhat = np.array(Ures)

        T = update_T(X,rho,Xhat,Lambda,error_measure,parameters)

        Lambda = Lambda + rho*(T - Xhat)
        k+=1
        obj,res= compute_objective_residual(X,Xhat,T,error_measure)
        objective.append(obj)
        residual.append(res)
        # Store elapsed time
        elapsed_time = time.time() - start_time
        time_history.append(elapsed_time)
        
   

   ## Plotting figures

    iterations = np.arange(len(objective))

    plt.figure(figsize=(12, 5))

    # --- Left plot: Objective ---
    plt.subplot(1, 2, 1)
    plt.semilogy(iterations, objective, linewidth=2)
    plt.title("Objective vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.grid(True)

        # --- Right plot: Residual ---
    plt.subplot(1, 2, 2)
    plt.semilogy(iterations, residual, color='orange', linewidth=2)
    plt.title("Residual vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("progress.png", dpi=200)
    plt.close()


    return T,A,B,C,objective,residual,time_history