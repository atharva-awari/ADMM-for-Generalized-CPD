## Kindly install pytensorlab package before running the code
import numpy as np
from pytensorlab import PolyadicTensor
from admm_tensors import admm_tensor

m = 100
n = 70
p = 30
r = 5
A0 = np.random.rand(m,r)
B0 = np.random.rand(n,r)
C0 = np.random.rand(p,r)
U = PolyadicTensor([A0,B0,C0])
X = np.array(U)

error_measure = 'KL divergence'

parameters = {'rho':1 , 'maxiter':100, 'maxtime': 10}
T,A,B,C,objective,residual,time_history,time_rec = admm_tensor(X,r,error_measure,parameters)
print("Final relative error=",objective[-1])
