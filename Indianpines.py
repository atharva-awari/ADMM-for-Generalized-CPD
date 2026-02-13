import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pytensorlab as ptl 
from admm_tensors import admm_tensor
from ao_admm_tensors import ao_admm_tensor
from pytensorlab import PolyadicTensor

X= loadmat('indianpines.mat')['X']
X_noisy = loadmat('indianpinesnoisy.mat')['X_noisy']

m,n,p = X.shape
eps=1e-8
r=70
print(X.shape)
print(X_noisy.shape)
band_id = 20
normX = ptl.frob(X)


## Initialization
Ainit = np.random.randn(m,r)
Binit = np.random.randn(n,r)
Cinit = np.random.randn(p,r)

## Define AO-ADMM for tensor parameters with frobenius norm
error_measure = 'KL divergence'
parameters = {'rho':1 , 'maxiter':np.inf, 'maxtime': 60, 'Ainit':Ainit, 'Binit':Binit, 'Cinit':Cinit, 'Xtrue': X}
T_ao,A_ao,B_ao,C_ao,objective_ao,residual_ao,time_history_ao,time_rec_ao = ao_admm_tensor(X_noisy,r,error_measure,parameters)
X_ao = PolyadicTensor([A_ao, B_ao, C_ao])
X_ao = np.array(X_ao)
print("AO-ADMM error=",ptl.frob(X - X_ao)/ normX)
print(objective_ao[-1])
## Define ADMM for tensor parameters with KL divergence
error_measure = 'KL divergence'
parameters = {'rho':1 , 'maxiter':np.inf, 'maxtime': 60, 'Ainit':Ainit, 'Binit':Binit, 'Cinit':Cinit, 'Xtrue': X}
T_admm,A_admm,B_admm,C_admm,objective_admm,residual_admm,time_history_admm, time_rec_admm = admm_tensor(X_noisy,r,error_measure,parameters)
X_admm = PolyadicTensor([A_admm, B_admm, C_admm])
X_admm = np.array(X_admm)
print("ADMM error=",ptl.frob(X - X_admm)/ normX)
print(objective_admm[-1])




# -----------------------------
# Prepare 2x2 grid for Band 29
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12,10))


# Top-left: clean
axes[0,0].imshow(X[:,:,band_id], cmap='gray')
axes[0,0].set_title('Clean Image')
axes[0,0].axis('off')

# Top-right: noisy
axes[0,1].imshow(X_noisy[:,:,band_id], cmap='gray')
axes[0,1].set_title('Noisy Image')
axes[0,1].axis('off')

# Bottom-left: Frobenius reconstruction
axes[1,0].imshow(X_ao[:,:,band_id], cmap='gray')
axes[1,0].set_title('Reconstruction AO-ADMM')
axes[1,0].axis('off')

# Bottom-right: KL reconstruction
axes[1,1].imshow(X_admm[:,:,band_id], cmap='gray')
axes[1,1].set_title('Reconstruction ADMM')
axes[1,1].axis('off')

plt.tight_layout()
plt.savefig("Indianpines.png", dpi=200)



## Save errors in a text file
np.savetxt("Indianpines_ao.txt", objective_ao)
np.savetxt("Indianpines_admm.txt", objective_admm)
np.savetxt("Indianpines_time_admm.txt", time_history_admm)
np.savetxt("Indianpines_time_ao.txt", time_history_ao)

## Plot comparison
# ----  Objective vs Time ----
plt.figure(figsize=(7, 5))
plt.semilogy(time_history_ao, objective_ao, label='AO-ADMM', linewidth=2)
plt.semilogy(time_history_admm, objective_admm, label='ADMM', linewidth=2)
plt.xlabel("Time (seconds)")
plt.ylabel("Objective value")
plt.title("Objective vs Time for Indian pines data with poisson noise")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("Indianpines_poisson.png", dpi=200)
plt.close()