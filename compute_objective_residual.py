import numpy as np

def compute_objective_residual(X, Xhat, T, error_measure):
    if error_measure == 'Frobenius':
        objective = np.linalg.norm(X - Xhat) / np.linalg.norm(X)
        residual = np.linalg.norm(T - Xhat) / np.linalg.norm(T)
    elif error_measure == 'L1':
        objective = np.sum(np.abs(X - Xhat)) / np.sum(np.abs(X))
        residual = np.sum(np.abs(T - Xhat)) / np.sum(np.abs(T))
    elif error_measure == 'KL divergence':
        eps = 0.0001

        # Ensure no zeros
        X_safe = np.maximum(eps,X)
        Xhat_safe = np.maximum(eps, Xhat)

        # Mean of all elements
        mu_X = np.mean(X_safe)

        # Construct X_est = mu_X * ones(m,n)
        m,n,p = X.shape
        X_est = mu_X * np.ones((m,n,p))

        # KL divergence matrix
        kl_tensor = X_safe * np.log(X_safe / Xhat_safe) - X_safe + Xhat_safe

        # Normalization KL term
        kl_est = X_safe * np.log(X_safe / X_est) - X_safe + X_est

        # Objective = normalized KL divergence
        objective = np.sum(kl_tensor) / np.sum(kl_est)

        # Residual = Frobenius norm ratio
        residual = np.linalg.norm(T - Xhat) / np.linalg.norm(T)
    else:
        raise ValueError(f"Unknown error_measure: {error_measure}")

    return objective, residual
