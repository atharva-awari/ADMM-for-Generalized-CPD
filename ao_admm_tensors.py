import numpy as np
import matplotlib.pyplot as plt
import pytensorlab as ptl
import time 


from pytensorlab import PolyadicTensor
from update_T import update_T
from compute_objective_residual import compute_objective_residual


def ao_admm_tensor(X,r,error_measure,parameters):
    m,n,p = X.shape
    rho_A = parameters.get('rho_A', 1.0)
    rho_B = parameters.get('rho_B', 1.0)
    rho_C = parameters.get('rho_C', 1.0)
    maxiter = parameters.get('maxiter', 1000)
    maxtime = parameters.get('maxtime', 10)
    A  = parameters.get('Ainit', np.random.randn(m,r))
    B  = parameters.get('Binit', np.random.randn(n,r))
    C  = parameters.get('Cinit', np.random.randn(p,r))
    inner_iter = parameters.get('inner_iter',10)
    T = X
    
    
    
    Lambda_A= np.zeros((m,n*p))
    Lambda_B= np.zeros((n,m*p))
    Lambda_C= np.zeros((p,m*n))
   
    X_A = ptl.tens2mat(X,0)
    X_B = ptl.tens2mat(X,1)
    X_C = ptl.tens2mat(X,2)
    ## For KL divergence T has to be initialized with positive entries where X is positive
    if error_measure == 'KL divergence':
       f = X > 0
       T[f] = np.maximum(np.finfo(float).eps, T[f])
    eps=1e-8
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
        
        KR_A = ptl.kr(B, C)              # (n*p, r)
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
        
        KR_B = ptl.kr(A, C)              # (m*p, r) depending on convention
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

        
        KR_C = ptl.kr(A, B)
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
        U = PolyadicTensor([A, B, C])
        Xhat = np.array(U)
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
