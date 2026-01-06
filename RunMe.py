import numpy as np
import tensorly as tl

from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from admm_tensors import admm_tensor
from ao_admm_tensors import ao_admm_tensor
m = 20
n = 15
p = 8
r = 3
A = np.random.randn(m,r)
B = np.random.randn(n,r)
C = np.random.randn(p,r)
X = cp_to_tensor((None,[A,B,C]))

error_measure = 'Frobenius'

parameters = {'rho':1 , 'maxiter':100, 'maxtime': 100}
T,A,B,C,objective,residual,time_history = ao_admm_tensor(X,r,error_measure,parameters)
#print("Final relative=",objective[-1])
Xhat_l1 = cp_to_tensor((None, [A,B,C]))
## CPD comparison
weights,factors = parafac(X,rank=r)
A1,B1,C1 = factors
Xhat = cp_to_tensor((weights,[A1,B1,C1]))
objective_cpd = np.linalg.norm(X - Xhat) / np.linalg.norm(X)
objective_cpd_l1 = np.sum(np.abs(X - Xhat)) / np.sum(np.abs(X))
objective_admm_l1 = np.sum(np.abs(X - Xhat_l1)) / np.sum(np.abs(X))
print("error_CPD=", objective_cpd)
print("error_CPD_l1=", objective_cpd_l1)
print("error_admm_l1=", objective_admm_l1)

