import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl 
import time 

from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
from update_T import update_T
from compute_objective_residual import compute_objective_residual
tl.set_backend('numpy')

def admm_tensor(X,r,error_measure,parameters):
    m,n,p = X.shape
    rho = parameters.get('rho', 1.0)
    maxiter = parameters.get('maxiter', 1000)
    maxtime = parameters.get('maxtime', 10)

    T = np.random.randn(m,n,p)
    A = np.random.randn(m,r)
    B = np.random.randn(n,r)
    C = np.random.randn(p,r)
    Lambda= np.zeros((m,n,p))
   
    ## For KL divergence T has to be initialized with positive entries where X is positive
    #T[X>0] = np.maximum(0.00001,T[X>0])
    #X[X==0]= 0.00001

    k=0
    start_time = time.time()
    Xhat = cp_to_tensor((None,[A,B,C]))
    objective = []
    residual  = []
    obj,res= compute_objective_residual(X,Xhat,T,error_measure)
    objective.append(obj)
    residual.append(res)
    while k<maxiter and (time.time() - start_time < maxtime):
        weights,factors = parafac(T + Lambda, rank=r)
        A,B,C = factors
        Xhat = cp_to_tensor((weights,factors))
        T = update_T(X,rho,Xhat,Lambda,error_measure,parameters)

        Lambda = Lambda + rho*(T - Xhat)
        k+=1
        obj,res= compute_objective_residual(X,Xhat,T,error_measure)
        objective.append(obj)
        residual.append(res)
        

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


    return T,A,B,C,objective,residual