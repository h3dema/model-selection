import random


def create_classes(num_classes=5):
    """
    Creates a class set C with num_classes classes.

    Parameters
    ----------
    num_classes : int, optional
        The number of classes in the class set C. Defaults to 5.

    Returns
    -------
    list
        A list of integers, each representing a class in the class set C.
    """
    return list(range(1, num_classes + 1))


# Create the dataset D with elements e_i
def create_dataset(size_dataset=1000):
    """
    Creates a dataset D with elements e_i.

    Parameters
    ----------
    size_dataset : int
        The size of the dataset to be created.

    Returns
    -------
    list
        A list of strings, each representing an element in the dataset D.

    """
    return [f"e_{i+1}" for i in range(size_dataset)]


# Assign a random class from C to each element in D
def assign_classes(D, C):
    """
    Assigns a random class from C to each element in D.

    Parameters
    ----------
    D : list
        A list of strings, each representing an element in the dataset D.
    C : list
        A list of integers, each representing a class in the class set C.

    Returns
    -------
    dict
        A dictionary where the keys are elements in D and the values are the randomly assigned classes from C.
    """
    return {e: random.choice(C) for e in D}


# Create subsets D_i without repeating elements
def create_subsets(D, subset_sizes):
    """
    Creates subsets D_i without repeating elements.

    Parameters
    ----------
    D : list
        A list of strings, each representing an element in the dataset D.
    subset_sizes : list
        A list of integers, each representing the size of a subset D_i.

    Returns
    -------
    dict
        A dictionary where the keys are the names of the subsets D_i and the values are the subsets themselves.
    """
    subsets = {}
    available_elements = D.copy()
    for i, size in enumerate(subset_sizes):
        if len(available_elements) < size:
            available_elements = D.copy()  # Reset if not enough elements
        subset = random.sample(available_elements, size)
        subsets[f"D_{i+1}"] = subset
        for e in subset:
            available_elements.remove(e)
    return subsets
