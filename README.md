# ADMM for Nonlinear Matrix Decompositions (NMD)

Given a data matrix $X \in \mathbb{R}^{m \times n}$ and a rank $r \ll \min(m,n)$, we solve the nonlinear matrix decomposition problem:

$$
\min_{T,\,W,\,H} \; d(X, f(T))
\quad \text{s.t.} \quad
T = WH
$$

where:
- $f(\cdot)$ is an elementwise nonlinear function,
- $d(\cdot,\cdot)$ is the loss,
- $T \in \mathbb{R}^{m \times n}$ is a rank-$r$ matrix,
- $W \in \mathbb{R}^{m \times r}$ and $H \in \mathbb{R}^{r \times n}$ are factors.

## Nonlinear Models \(f(T)\)

- **ReLU**  
  $$
  f(T) = \max(0,\, T)
  $$

- **CSF (elementwise square)**  
  $$
  f(T) = T^{.2} = T \odot T
  $$

- **Min–Max**  
  With bounds $a \le b$:
  $$
  f(T) = \min\!\big(b,\, \max(a,\, T)\big)
  $$

- **Modulus**  
  $$
  f(T) = |T|
  $$


## Loss Functions \(d(X, f(T))\)

- **Frobenius norm**
  $$
  \lVert X - f(T) \rVert_F
  \;=\;
  \sqrt{\sum_{i=1}^m \sum_{j=1}^n \big( X_{ij} - [f(T)]_{ij} \big)^2}
  $$

- **\(L_1\) norm**
  $$
  \lVert X - f(T) \rVert_1
  \;=\;
  \sum_{i=1}^m \sum_{j=1}^n \left|\, X_{ij} - [f(T)]_{ij} \,\right|
  $$

- **Kullback–Leibler (KL) divergence**  
  For nonnegative scalars $x,y \ge 0$:
  $$
  d_{\mathrm{KL}}(x,y) =
  \begin{cases}
    x \log\!\left(\dfrac{x}{y}\right) - x + y, & x>0,\\[6pt]
    y, & x=0
  \end{cases}
  $$
  Aggregated over the matrix:
  $$
  KL\!\big(X,\, f(T)\big)
  \;=\;
  \sum_{i,j} d_{\mathrm{KL}}\!\Big( X_{ij},\, [f(T)]_{ij} \Big)
  $$
