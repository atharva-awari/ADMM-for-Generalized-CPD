import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl 
import pytensorlab as ptl
import time 

from tensorly.cp_tensor import cp_to_tensor
from update_T import update_T
from compute_objective_residual import compute_objective_residual
tl.set_backend('numpy')

def ao_admm_tensor(X,r,error_measure,parameters):
    m,n,p = X.shape
    # rho = parameters.get('rho', 1.0)
    maxiter = parameters.get('maxiter', 1000)
    maxtime = parameters.get('maxtime', 10)
    inner_iter = 10
    T = X
    A = np.random.randn(m,r)
    B = np.random.randn(n,r)
    C = np.random.randn(p,r)
    
    Lambda_A= np.zeros((m,n*p))
    Lambda_B= np.zeros((n,m*p))
    Lambda_C= np.zeros((p,m*n))

    rho_A =0.1
    rho_B =0.1
    rho_C =0.1
   
    X_A = ptl.tens2mat(X,0)
    X_B = ptl.tens2mat(X,1)
    X_C = ptl.tens2mat(X,2)
    ## For KL divergence T has to be initialized with positive entries where X is positive
    #T[X>0] = np.maximum(0.00001,T[X>0])
    #X[X==0]= 0.00001
    eps = 1e-8
    k=0
    start_time = time.time()
    Xhat = cp_to_tensor((None,[A,B,C]))
    objective = []
    residual  = []
    obj,res= compute_objective_residual(X,Xhat,T,error_measure)
    objective.append(obj)
    residual.append(res)
    while k<maxiter and (time.time() - start_time < maxtime):
        
        KR_A = ptl.kr(C, B)              # (n*p, r)
        G_A = (np.transpose(C) @ C) * (np.transpose(B) @ B) # (r, r)
        G_A = G_A + eps * np.eye(r) # Ridge regularizer
        T_A  = ptl.tens2mat(T, 0)        # (m, n*p)
        for it in range(inner_iter):
           ## Update A and T_A keeping B and C fixed 
            M = (T_A + (1.0/rho_A) * Lambda_A) @ KR_A   # (m, r)
            A = np.transpose(np.linalg.solve(G_A, np.transpose(M)))  # Update A
            Xhat = A@np.transpose(KR_A)
            T_A = update_T(X_A,rho_A,Xhat,Lambda_A,error_measure,parameters) # Update T_A

            Lambda_A = Lambda_A + rho_A * (T_A - A @ KR_A.T) #Update Dual variable
        
        KR_B = ptl.kr(C, A)              # (m*p, r) depending on convention
        G_B = (np.transpose(C) @ C) * (np.transpose(A) @ A)
        G_B = G_B + eps*np.eye(r)
        T_B  = ptl.tens2mat(T, 1)        # (n, m*p)
        ##Update B and T_B keeping A and C fixed
        for it in range(inner_iter): 
            M = (T_B + (1.0/rho_B) * Lambda_B) @ KR_B
            B = np.transpose(np.linalg.solve(G_B, np.transpose(M)))
            Xhat = B@np.transpose(KR_B)
            T_B = update_T(X_B,rho_B,Xhat,Lambda_B,error_measure,parameters) # Update T_B
            
            Lambda_B = Lambda_B + rho_B * (T_B - B @ KR_B.T) #Update Dual variable

        
        KR_C = ptl.kr(B, A)
        G_C = (np.transpose(B) @ B) * (np.transpose(A) @ A)
        G_C = G_C + eps*np.eye(r)
        T_C  = ptl.tens2mat(T, 2)
        ##Update C and T_C keeping A and B fixed 
        for it in range(inner_iter):  
            M = (T_C + (1.0/rho_C) * Lambda_C) @ KR_C
            C = np.transpose(np.linalg.solve(G_C, np.transpose(M)))
            Xhat = C@np.transpose(KR_C)
            T_C = update_T(X_C,rho_C,Xhat,Lambda_C,error_measure,parameters) # Update T_C
            
            Lambda_C = Lambda_C + rho_C * (T_C - C @ KR_C.T) #Update Dual variable
                    
        
        k+=1
        Xhat = cp_to_tensor((None,[A,B,C]))
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