import numpy as np
def update_T(X, rho, Xhat, Lambda, error_measure, parameters):
    rho = parameters.get('rho', 1.0)

    if error_measure == 'Frobenius':
        T = (2 * X + rho * (Xhat - Lambda)) / (rho + 2)
    elif error_measure == 'L1':
        t1star = (rho*(Xhat - Lambda) + 1 ) / rho
        t2star = (rho*(Xhat - Lambda) - 1 ) / rho
        T = np.maximum(np.minimum(t1star, X), t2star)
    elif error_measure == 'KL divergence':
        T = np.zeros_like(X)
        p = X>0
        q = X==0
        T[q] = (rho*Xhat[q] - rho*Lambda[q] -1) / rho
        D = np.sqrt((1 - rho*Xhat + Lambda)**2 + 4*rho*X)
        T[p] = ((-1 + rho*Xhat[p] - Lambda[p]) + D[p] ) / (2*rho)
    else:
        raise ValueError(f"Unknown error_measure: {error_measure}")

    return T
