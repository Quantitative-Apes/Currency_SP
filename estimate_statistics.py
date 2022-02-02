import numpy as np
from numba import jit

#@jit
def est_log_normal(P, dt):
    """Estimate mean and covariance matrix given that prices in P follow a log-normal distribution"""
    # dt is time of each period in years
    log_returns = (np.log(P[1:,:])-np.log(P[0:-1, :]))
    var = np.var(log_returns, ddof=1, axis=0)/dt
    Rho = np.corrcoef(log_returns, rowvar=False)
    nu = np.mean(log_returns, axis=0)/dt
    Sigma = np.sqrt(np.diag(var)) @ Rho @ np.sqrt(np.diag(var))

    return (nu, Sigma)