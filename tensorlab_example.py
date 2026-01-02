import numpy as np
import pytensorlab as tl

# Create a 3D tensor
X = np.random.rand(4, 3, 2)

print(X.shape)

# Mode-0 unfolding
X0 = tl.tens2mat(X)

print(X0.shape)
print(X)
print(X0)