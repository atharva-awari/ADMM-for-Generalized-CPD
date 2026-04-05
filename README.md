# ADMM for Generalized Tensor Decomposition

Given a tensor $\mathcal{X} \in \mathbb{R}^{m \times n \times p}$ and a rank $r \ll \min(m,n,p)$, we solve the Generalized Canononical Polyadic (CP/PARAFAC) decomposition problem:

$$
\min_{\T,\,A,\,B, \C} \; d(\mathcal{X}, \mathcal{T})
\quad \text{s.t.} \quad
T = [A,B,C]
$$

where:
- $d(\cdot,\cdot)$ is the loss-function,
- $T \in \mathbb{R}^{m \times n \times p}$ is a rank-$r$ tensor,
- $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{n \times r}$ and  $C \in \mathbb{R}^{p \times r}$are factor matrices of the CP.


## Loss Functions $(d(\mathcal{X}, mathcal{T} \)$

- **Frobenius norm**
  $$
  \lVert \mathcal{X} - \mathcal{T} \rVert_F
  \;=\;
  \sqrt{\sum_{i=1}^m \sum_{j=1}^n \big( \mathcal{X}_{ij} - \mathcal{T}_{ij} \big)^2}
  $$

- **\(L_1\) norm**
  $$
  \lVert \mathcal{X} - \mathcal{T} \rVert_1
  \;=\;
  \sum_{i=1}^m \sum_{j=1}^n \left|\, \mathcal{X}_{ij} - \mathcal{T}_{ij} \,\right|
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
  KL\!\big(\mathcal{X}, \mathcal{T}\big)
  \;=\;
  \sum_{i,j} d_{\mathrm{KL}}\!\Big( \mathcal{X}_{ij},\, \mathcal{T}_{ij} \Big)
  $$
