
import numpy as np

def ccf2acf(ccf):
    """
    Compute the ACF of each CCF (eq. 3 in [1])

    Args:
        ccf (array, Nobs x Nrv) Cross-correlation function for each observation
    Returns:
        acf (array, Nobs x Nrv) Autocorrelation function of each CCF

    References:
        [1] Collier Cameron et al. (2020) arxiv:2011.00018
    """
    ccf = np.atleast_2d(ccf)

    # number of CCF "pixels" or velocity bins
    Nrv = ccf.shape[1]

    # normalize by the mean CCF of each observation
    ccf_norm = ccf / ccf.mean(axis=1, keepdims=True)

    # to normalize the ACF to 1
    acf0 = (ccf**2).sum(axis=1)

    # calculate ACF
    acf = []
    for i in range(Nrv):
        ccf_shifted = np.roll(ccf_norm, i, axis=1)
        acf.append(np.sum(ccf_norm * ccf_shifted, axis=1) / acf0)

    # re-normalize the ACF
    acf = np.array(acf) / np.mean(acf, axis=0)

    return acf.T
