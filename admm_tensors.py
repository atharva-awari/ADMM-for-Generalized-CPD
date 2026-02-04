import numpy as np
import matplotlib.pyplot as plt
import pytensorlab as ptl
import time 

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
    time_rec = 0
    eps = 1e-8
    while k<maxiter and (time.time() - start_time < maxtime):

        ## Update A using ALS
        KR_A = ptl.kr(B, C)              # (n*p, r)
        G_A = (np.transpose(C) @ C) * (np.transpose(B) @ B) # (r, r)
        G_A = G_A + eps * np.eye(r) # Ridge regularizer
        T_A  = ptl.tens2mat(T, 0)        # (m, n*p)
        Lambda_A = ptl.tens2mat(Lambda,0)
        M = (T_A + (1.0/rho) * Lambda_A) @ KR_A   # (m, r)
        A = np.transpose(np.linalg.solve(G_A, np.transpose(M)))  # Update A

        ## Update B using ALS
        KR_B = ptl.kr(A, C)              # (m*p, r) depending on convention
        G_B = (np.transpose(C) @ C) * (np.transpose(A) @ A)
        G_B = G_B + eps*np.eye(r)
        T_B  = ptl.tens2mat(T, 1)        # (n, m*p)
        Lambda_B = ptl.tens2mat(Lambda,1)
        M = (T_B + (1.0/rho) * Lambda_B) @ KR_B
        B = np.transpose(np.linalg.solve(G_B, np.transpose(M)))

        ## Update C using ALS
        KR_C = ptl.kr(A, B)
        G_C = (np.transpose(B) @ B) * (np.transpose(A) @ A)
        G_C = G_C + eps*np.eye(r)
        T_C  = ptl.tens2mat(T, 2) 
        Lambda_C = ptl.tens2mat(Lambda,2)
        M = (T_C + (1.0/rho) * Lambda_C) @ KR_C
        C = np.transpose(np.linalg.solve(G_C, np.transpose(M)))
        
        U = PolyadicTensor([A, B, C])
        Xhat = np.array(U)
        ## Update T scalar-wise
        T = update_T(X,rho,Xhat,Lambda,error_measure,parameters)
        
        ## Update dual variable
        Lambda = Lambda + rho*(T - Xhat)
        k+=1
        obj,res= compute_objective_residual(X,Xhat,T,error_measure)
        objective.append(obj)
        residual.append(res)
        # Store elapsed time
        elapsed_time = time.time() - start_time
        time_history.append(elapsed_time)
        if time_rec == 0 and obj<= 1e-6:
            time_rec = elapsed_time
            
        
        

           
        
   

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


    return T,A,B,C,objective,residual,time_history,time_rec