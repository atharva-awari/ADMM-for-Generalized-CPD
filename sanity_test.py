import numpy as np
import pytensorlab as ptl
# Small test
Z=3  
J=4
K=5 
R=2
A0 = np.random.rand(Z, R)
B0 = np.random.rand(J, R)
C0 = np.random.rand(K, R)

# Construct tensor from true factors
T0 = np.einsum("ir,jr,kr->ijk", A0, B0, C0)

T1 = ptl.tens2mat(T0, 0)
KR = ptl.kr(C0, B0)

err = np.linalg.norm(T1 - A0 @ KR.T) / np.linalg.norm(T1)
print("Relative error:", err)