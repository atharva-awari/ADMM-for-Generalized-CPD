import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl 

from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor

X = np.random.randn(4,4,4)
cp_rank=2

weights,factors = parafac(X, rank=cp_rank)

A,B,C = factors

print("A=" ,A, "B=", B, "C=",C)