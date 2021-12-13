from math import tau
import numpy as np
from scipy.stats import norm as Gaussian


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


def scalpels(ccf, rv, rverr, k, ivw=True, return_usp=False):
    """
    Calculate the singular value decomposition of the ACF and project the
    measured radial velocities onto the first k principal components.

    Args:
        ccf (array, Nobs x Nrv) Cross-correlation function for each observation
        rv (array, Nobs) Measured radial velocities
        rverr (array, Nobs) Radial velocity uncertainties (used if ivw=True)
        k (int) Number of basis vectors to project onto
        ivw (bool) Subtract inverse-variance weighted average?
        return_usp (bool) Return the output from SVD decomposition of the ACF
    Returns:
        v_obs, v_shape, v_shift (arrays, Nobs)
            Observed RVs (minus average), shape-driven RVs, shift-driven RVs
        (u, s, p) (tuple of arrays)
            Output from SVD decomposition of the ACF, if `return_usp` is True

    References:
        [1] Collier Cameron et al. (2020) arxiv:2011.00018
    """
    if ivw:
        # subtract inverse-variance weighted average
        invvar = 1 / rverr**2
        v_obs = rv - np.average(rv, weights=invvar)
    else:
        v_obs = rv - rv.mean()

    # calculate ACF
    acf = ccf2acf(ccf)
    # SVD decomposition (eq. 5 [1])
    u, s, p = np.linalg.svd(acf, full_matrices=False)
    # U_A, shape (Nobs x Nobs)
    # S_A, shape (Nobs x 1)
    # P_A, shape (Nobs x Nrv)

    # vector of response factors (eq. 6 [1])
    alpha = np.dot(u.T, v_obs)

    # sort the elements of alpha in order of descending absolute value
    # (see section 3.3 [1])
    ind = np.argsort(np.abs(alpha))[::-1]
    u_sort = u[:, ind]
    alpha_sort = alpha[ind]

    # use only the first k basis vectors
    u_sort = u_sort[:, :k]
    alpha_sort = alpha_sort[:k]

    # calculate v∥ (shape-driven)
    v_shape = np.dot(u_sort, alpha_sort)
    # calculate v⟂ (shift-driven)
    v_shift = v_obs - v_shape

    if return_usp:
        return v_obs, v_shape, v_shift, (u, s, p)

    return v_obs, v_shape, v_shift


def BIC(k, v_shift, rverr=None):
    """
    Calculate the Bayesian Information Criterium (BIC) for a SCALPELS model
    that includes k principal components.

    Args:
        k (int) Number of principal components used in the decomposition
        v_shift (array, Nobs) Shift-driven RVs obtained from SCALPELS
        rverr (array, Nobs, optional) Radial velocity uncertainties
    Returns:
        BIC (float)
            The BIC value for the SCALPELS model with k principal components

    References:
        [1] Collier Cameron et al. (2020) arxiv:2011.00018
    """
    if rverr is None:
        rverr = np.ones_like(v_shift)
    # the log-likelihood (using scipy.stats.norm for clarity)
    logL = Gaussian(v_shift.mean(), rverr).logpdf(v_shift).sum()
    # penalty for subtracting mean and adding k basis vectors
    pen = (k + 1) * np.log(v_shift.size)
    # return the BIC (see e.g. Wikipedia)
    return -2 * logL + pen


def best_k(ccf, rv, rverr, ivw=True, BIC_threshold=10):
    """
    Estimate the number of principal components to use for SCALPELS.

    Args:
        ccf (array, Nobs x Nrv) Cross-correlation function for each observation
        rv (array, Nobs) Measured radial velocities
        rverr (array, Nobs) Radial velocity uncertainties
        ivw (bool) Subtract inverse-variance weighted average?
        BIC_threshold (float) The theshold for a "significant" decrease in BIC
    Returns:
        k (int) "Optimal" number of principal components
    Raises:
        ValueError, if the best estimate of k is equal to the total number of
        basis vectors

    References:
        [1] Collier Cameron et al. (2020) arxiv:2011.00018
    """
    n = ccf.shape[0]
    best_bic = np.inf
    # try all values of k
    for k in range(n + 1):
        # run SCALPELS
        *_, shift = scalpels(ccf, rv, rverr, k, ivw=ivw)
        # estimate BIC
        bic = BIC(k, shift, rverr)
        # if BIC improved by more than BIC_threshold, continue
        if bic < best_bic and (best_bic - bic) > BIC_threshold:
            best_bic = bic
        # otherwise, return previous k
        else:
            return k - 1
    # if we got here, something went wrong?
    raise ValueError('Estimate of k is too high?')


def scalpels_planet(ccf, time, rv, rverr, k, period, ivw=True):
    """
    Calculate the singular value decomposition of the ACF and project the
    measured radial velocities onto the first k principal components including
    a simultaneous sinusoidal fit (see section 4.2 and appendix C2 in [1]).

    Args:
        ccf (array, Nobs x Nrv) Cross-correlation function for each observation
        time (array, Nobs) Times at which radial velocities were observed
        rv (array, Nobs) Measured radial velocities
        rverr (array, Nobs) Radial velocity uncertainties
        k (int) Number of basis vectors to project onto
        period (float) Orbital period of the planet (circular orbit)
        ivw (bool) Subtract inverse-variance weighted average?
    Returns:
        v_obs, v_shape, v_shift, v_orb (arrays, Nobs)
            Observed RVs (minus average), shape-driven RVs, shift-driven RVs,
            and orbital RVs.
        θ (array, 2)
            Best-fit planet coefficient pair (A1,B1)

    References:
        [1] Collier Cameron et al. (2020) arxiv:2011.00018
    """
    if ivw:
        # subtract inverse-variance weighted average
        invvar = 1 / rverr**2
        v_obs = rv - np.average(rv, weights=invvar)
    else:
        v_obs = rv - rv.mean()

    # calculate ACF
    acf = ccf2acf(ccf)
    # SVD decomposition (eq. 5 [1])
    u, s, p = np.linalg.svd(acf, full_matrices=False)
    # U_A, shape (Nobs x Nobs)
    # S_A, shape (Nobs x 1)
    # P_A, shape (Nobs x Nrv)

    # vector of response factors (eq. 6 [1])
    alpha = np.dot(u.T, v_obs)

    # sort the elements of alpha in order of descending absolute value
    # (see section 3.3 [1])
    ind = np.argsort(np.abs(alpha))[::-1]
    u_sort = u[:, ind]
    alpha_sort = alpha[ind]

    # use only the first k basis vectors
    u_sort = u_sort[:, :k]
    alpha_sort = alpha_sort[:k]

    # up to here, it was just like scalpels

    # compute F and concatenate it to U_A (see section 4.2 [1])
    F = np.c_[np.cos(time * tau / period), np.sin(time * tau / period)]
    A = np.c_[F, u_sort]

    #! skipping steps iii - vii in appendix C2
    Σinv = np.diag(1 / rverr**2)

    # solve the least-squares problem (eq. 11 [1])
    temp = np.dot(A.T, Σinv)
    a = temp.dot(A)
    b = temp.dot(v_obs)
    θ = np.linalg.solve(a, b)

    # calculate v∥ (shape-driven)
    v_shape = np.dot(u_sort, θ[2:])
    # calculate v_orb (planet)
    v_orb = np.dot(θ[:2], F.T)
    # calculate v⟂ (shift-driven)
    v_shift = v_obs - v_shape

    return v_obs, v_shape, v_shift, v_orb, θ[:2]
