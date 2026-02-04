## Kindly install pytensorlab package before running the code
import numpy as np
import pytensorlab as ptl
from pytensorlab import PolyadicTensor
from admm_tensors import admm_tensor
from ao_admm_tensors import ao_admm_tensor
m = 100
n = 70
p = 30
r = 5
A0 = np.random.randn(m,r)
B0 = np.random.randn(n,r)
C0 = np.random.randn(p,r)
U = PolyadicTensor([A0,B0,C0])
X = np.array(U)

error_measure = 'Frobenius'

parameters = {'rho':1 , 'maxiter':100, 'maxtime': 10}
T,A,B,C,objective,residual,time_history,time_rec = admm_tensor(X,r,error_measure,parameters)
print("Final relative error=",objective[-1])
