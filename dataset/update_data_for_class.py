import random


def update_subset_for_class(D, subset, c, class_map, evolution_obj):
    """
    Updates a subset based on the current state from AnimalDatasetEvolution for class c.

    Parameters
    ----------
    D : list
        Full dataset of elements.
    subset : list
        Current subset D_i to be updated.
    c : any
        The class label to operate on.
    class_map : dict
        A mapping from element to class, e.g., {e_1: 1, e_2: 2, ...}
    evolution_obj : AnimalDatasetEvolution
        An object that returns the next state: 'KEEP', 'INSERT', or 'REMOVE'.

    Returns
    -------
    list
        Updated subset.
    """
    state = evolution_obj.step()
    print("Next state:", state)

    if state == 'KEEP':
        return subset  # No change

    elif state == 'INSERT':
        # Find candidates in D with class c not already in subset
        candidates = [e for e in D if class_map[e] == c and e not in subset]
        if candidates:
            e_new = random.choice(candidates)
            subset.append(e_new)
        return subset

    elif state == 'REMOVE':
        # Find elements in subset with class c
        candidates = [e for e in subset if class_map[e] == c]
        if len(candidates) > 1:
            e_remove = random.choice(candidates)
            subset.remove(e_remove)
        return subset

    else:
        # should never reach here
        raise ValueError(f"Unknown state: {state}")
