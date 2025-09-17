import numpy as np


def zipf_pmf(N: int, alpha: float):
    """
    Compute the probability mass function (PMF) of a Zipf distribution with N items and parameter alpha.

    Parameters
    ----------
    N : int
        Number of items.
    alpha : float
        Parameter of the Zipf distribution.

    Returns
    -------
    ks : ndarray of shape (N,)
        Indices of the items.
    pmf : ndarray of shape (N,)
        Probability mass function, i.e., the probability of each item.

    Notes
    -----
    The PMF is defined as pmf(k) = k**(-alpha) / Z, where Z is the normalization constant Z = sum(k**(-alpha)) over k from 1 to N.
    """
    assert alpha > 1, "alpha must be greater than 1"

    ks = np.arange(1, N + 1)
    weights = ks ** (-alpha)
    Z = weights.sum()
    return ks, weights / Z


def zipf_sample(N: int, alpha: float, size: int = 1):
    """
    Sample from a Zipf distribution with N items and parameter alpha.

    Parameters
    ----------
    N : int
        Number of items.
    alpha : float
        Parameter of the Zipf distribution.
    size : int, optional
        Number of samples to draw. Default is 1.

    Returns
    -------
    samples : ndarray of shape (size,)
        Samples drawn from the Zipf distribution.
    """
    assert alpha > 1, "alpha must be greater than 1"
    assert size > 0, "size must be positive"

    ks = np.arange(1, N + 1)
    weights = ks ** (-alpha)
    probs = weights / weights.sum()
    return np.random.choice(ks, size=size, p=probs)
