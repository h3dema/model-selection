import math
import random
from scipy.stats import norm


class Model:
    def __init__(self, energy, confidence, size, stdev, classes):
        """
        Initializes a Model object with the given parameters.

        Parameters
        ----------
        energy : float
            The energy of the model.
        confidence : float
            The base confidence of the model.
        size : float
            The size of the model.
        stdev : float or list[float]
            The standard deviation of the model. If a float, it is used
            for all classes. If a list, it must have the same length as classes.
        classes : list[int]
            The classes of the model.

        Raises
        ------
        ValueError
            If stdev is not a float or a list with the same length as classes.
        """
        self.energy = energy
        self.confidence_base = confidence
        self.size = size
        self.classes = classes

        # Normalize stdev input
        if isinstance(stdev, (int, float)):
            # Same stdev for all classes
            self.stdev_map = {c: stdev for c in classes}
        elif isinstance(stdev, (list, tuple)) and len(stdev) == len(classes):
            # Class-specific stdevs
            self.stdev_map = {c: s for c, s in zip(classes, stdev)}
        else:
            raise ValueError("stdev must be a float or a list with the same length as classes")

    def confidence(self, c):
        """
        Computes the confidence of a given class c.

        The confidence is computed by sampling from a normal distribution with
        mean 0 and standard deviation stdev, and then computing the probability
        density of the sampled value under the same normal distribution. The
        computed probability density is then normalized by the maximum possible
        value of the PDF.

        Parameters
        ----------
        c : int
            The class for which to compute the confidence.

        Returns
        -------
        float
            The normalized confidence of class c.

        Raises
        ------
        ValueError
            If c is not found in the model's class set, or if the standard deviation is not positive.
        """
        if c not in self.stdev_map:
            raise ValueError(f"Class {c} not found in model's class set")

        stdev = self.stdev_map[c]
        if stdev <= 0:
            raise ValueError("Standard deviation must be positive")

        # Sample from N(0, stdev)
        sample = random.gauss(0, stdev)

        # Compute probability density of the sampled value under N(0, stdev)
        pdf = norm(loc=0, scale=stdev).pdf(sample)

        # Normalize by the maximum PDF value
        max_pdf = 1 / (stdev * math.sqrt(2 * math.pi))
        normalized_confidence = pdf / max_pdf

        return normalized_confidence

    def __repr__(self):
        return f"Model(energy={self.energy}, size={self.size})"


def generate_models(M: int, C: list, e_max=10, s_max=10, seed=42):
    """
    Generates a list of M models with random attributes.

    Each model has random energy, size, and confidence between 0.5 and 1.0.
    The standard deviation is either a single uniform value between 0.0 and 0.3,
    or a list of per-class standard deviations with the same uniform distribution.

    Parameters
    ----------
    M : int
        The number of models to generate.
    C : list
        The list of classes.
    e_max : float, optional
        The maximum energy of a model. Defaults to 10.
    s_max : float, optional
        The maximum size of a model. Defaults to 10.
    seed : int, optional
        The random seed to use. Defaults to 42.

    Returns
    -------
    list
        A list of M models with random attributes.
    """
    random.seed(seed)
    epsilon = 1e-6  # Small value to avoid zero

    models = []
    for i in range(M):
        energy = random.uniform(epsilon, e_max)
        size = random.uniform(epsilon, s_max)
        confidence = random.uniform(0.5, 1)

        # Generate either a single stdev or per-class stdevs
        if random.random() < 0.5:
            stdev = random.uniform(0.0, 0.3)  # uniform stdev
        else:
            stdev = [random.uniform(0.0, 0.3) for _ in C]  # per-class stdevs

        model = Model(energy=energy, confidence=confidence, size=size, stdev=stdev, classes=C)
        models.append(model)

    return models
