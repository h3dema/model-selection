import numpy as np


class GilbertModel:
    """
        A simple implementation of the Gilbert model for bursty errors.
        The model is a two-state Markov chain, with states CORRECT and CORRUPT.
        The transition probabilities are given by p_cc, p_cp, p_pc, and p_pp.

    """

    def __init__(self,
                 p_cc=0.9913,  # CORRECT → CORRECT
                 p_cp=0.0087,  # CORRECT → CORRUPT
                 p_pc=0.1491,  # CORRUPT → CORRECT
                 p_pp=0.8509): # CORRUPT → CORRUPT

        """
        Initialize the Gilbert model.

        Parameters
        ----------
        p_cc : float, optional
            The probability of staying in the CORRECT state, by default 0.9913.
        p_cp : float, optional
            The probability of transitioning from the CORRECT to the CORRUPT state, by default 0.0087.
        p_pc : float, optional
            The probability of transitioning from the CORRUPT to the CORRECT state, by default 0.1491.
        p_pp : float, optional
            The probability of staying in the CORRUPT state, by default 0.8509.
        """
        self.p_cc = p_cc
        self.p_cp = p_cp
        self.p_pc = p_pc
        self.p_pp = p_pp
        self.state = 0  # Start in CORRECT (0)

    def step(self):
        """
        Take a single step in the Gilbert model.

        Returns
        -------
        int
            The new state of the model, either 0 (CORRECT) or 1 (CORRUPT).
        """
        if self.state == 0:  # CORRECT
            self.state = 0 if np.random.rand() < self.p_cc else 1
        else:  # CORRUPT
            self.state = 1 if np.random.rand() < self.p_pp else 0
        return self.state

    def sample(self, N: int = 1):
        """
        Sample N steps from the Gilbert model.

        Parameters
        ----------
        N : int, optional
            The number of steps to sample, by default 1.

        Returns
        -------
        states : ndarray of shape (N,)
            The sampled states of the model, either 0 (CORRECT) or 1 (CORRUPT).
        """
        assert N > 0, "N must be a positive integer"
        states = np.zeros(N, dtype=int)
        for i in range(N):
            states[i] = self.step()
        return states


# Calculate sampled transition probabilities
def compute_transition_probabilities(samples):
    """
    Compute transition probabilities from a given set of samples.

    Parameters
    ----------
    samples : ndarray of shape (N,)
        The sampled states of the model, either 0 (CORRECT) or 1 (CORRUPT).

    Returns
    -------
    probs : dict
        A dictionary containing the transition probabilities, with keys (i, j) and values p_ij.
    """
    transitions = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    counts = {0: 0, 1: 0}

    for i in range(N - 1):
        curr, next_ = samples[i], samples[i + 1]
        transitions[(curr, next_)] += 1
        counts[curr] += 1

    probs = {}
    for (i, j), count in transitions.items():
        probs[(i, j)] = count / counts[i] if counts[i] > 0 else 0

    return probs
