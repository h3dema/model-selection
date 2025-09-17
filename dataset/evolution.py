import random
import numpy as np
import matplotlib.pyplot as plt


class DatasetEvolution:
    def __init__(self, transition_probs, initial_state='KEEP', seed=None):
        """
        transition_probs: dict with keys like 'kk', 'ki', 'kr', 'ik', ..., 'rr'
        initial_state: one of 'KEEP', 'INSERT', 'REMOVE'
        """
        if seed is not None:
            random.seed(seed)

        self.states = ['KEEP', 'INSERT', 'REMOVE']
        self.state = initial_state.upper()

        # Validate and store transition probabilities
        self.transitions = {
            'KEEP': {
                'KEEP': transition_probs['kk'],
                'INSERT': transition_probs['ki'],
                'REMOVE': transition_probs['kr']
            },
            'INSERT': {
                'KEEP': transition_probs['ik'],
                'INSERT': transition_probs['ii'],
                'REMOVE': transition_probs['ir']
            },
            'REMOVE': {
                'KEEP': transition_probs['rk'],
                'INSERT': transition_probs['ri'],
                'REMOVE': transition_probs['rr']
            }
        }

        # Ensure each row sums to 1
        for origin in self.states:
            total = sum(self.transitions[origin].values())
            if not abs(total - 1.0) < 1e-6:
                raise ValueError(f"Transition probabilities from {origin} must sum to 1. Got {total}")

    def step(self):
        """Advance one step in the Markov chain and return the new state."""
        probs = self.transitions[self.state]
        next_state = random.choices(
            population=list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]
        self.state = next_state
        return self.state

    def simulate(self, T):
        states = []
        for _ in range(T):
            states.append(self.step())
        return states


def compute_balanced_transition_matrix(target_keep_ratio=0.95, tolerance=1e-6):
    # States: KEEP (0), INSERT (1), REMOVE (2)
    # We want π = [π_k, π_i, π_r] with π_k ≈ target_keep_ratio

    # Assume symmetric behavior for INSERT and REMOVE
    # We'll solve for a matrix P such that πP = π and π_k = target_keep_ratio

    # Let’s define π
    pi_k = target_keep_ratio
    pi_i = (1 - pi_k) / 2
    pi_r = (1 - pi_k) / 2
    pi = np.array([pi_k, pi_i, pi_r])

    # Define symbolic transition matrix P with unknowns
    # We'll fix most values and solve for a few to satisfy πP = π
    # Initial structure:
    # From KEEP: [p_kk, p_ki, p_kr]
    # From INSERT: [p_ik, p_ii, p_ir]
    # From REMOVE: [p_rk, p_ri, p_rr]

    # We'll fix transitions from KEEP
    p_kk = 0.97
    p_ki = 0.02
    p_kr = 0.01

    # We'll assume INSERT and REMOVE behave similarly
    # Let INSERT → KEEP = REMOVE → KEEP = x
    # Let INSERT → INSERT = REMOVE → REMOVE = y
    # Then INSERT → REMOVE = REMOVE → INSERT = 1 - x - y

    # Solve for x and y such that:
    # π_i * [x, y, 1 - x - y] + π_r * [x, 1 - x - y, y] + π_k * [p_kk, p_ki, p_kr] = π

    # Let’s choose x = 0.90, y = 0.05
    x = 0.90
    y = 0.05
    z = 1 - x - y  # = 0.05

    P = np.array([
        [p_kk, p_ki, p_kr],
        [x, y, z],
        [x, z, y]
    ])

    # Compute resulting stationary distribution
    eigvals, eigvecs = np.linalg.eig(P.T)
    stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stationary = stationary[:, 0]
    stationary = stationary / stationary.sum()

    # Check if π_KEEP is close to target
    keep_error = abs(stationary[0] - target_keep_ratio)

    # Check net size change: INSERT → KEEP vs REMOVE → KEEP
    insert_rate = stationary[1] * P[1][0]
    remove_rate = stationary[2] * P[2][0]
    net_change = insert_rate - remove_rate

    if keep_error < tolerance and abs(net_change) < tolerance:
        print("✅ Transition matrix satisfies constraints.")
    else:
        print("⚠️ Transition matrix does not fully satisfy constraints.")
        print(f"\tKEEP ratio error: {keep_error:.6f}")
        print(f"\tNet size change per step: {net_change:.6f}")

    # Return labeled matrix
    states = ['KEEP', 'INSERT', 'REMOVE']
    transition_matrix = {
        from_state: {
            to_state: P[i][j]
            for j, to_state in enumerate(states)
        }
        for i, from_state in enumerate(states)
    }

    return transition_matrix


def flatten_transition_matrix(matrix):
    flat = {}
    for from_state, transitions in matrix.items():
        for to_state, prob in transitions.items():
            key = from_state[0].lower() + to_state[0].lower()
            flat[key] = prob
    return flat



def plot_state_transitions(numeric_states, xmin=None, xmax=None, filename="animal_dataset_evolution.pdf"):
    time = list(range(len(numeric_states)))

    plt.figure(figsize=(12, 4))
    plt.step(time, numeric_states, where='post')
    plt.yticks([0, 1, 2], ['KEEP', 'INSERT', 'REMOVE'])
    plt.xlabel('Timestep')
    plt.ylabel('State')
    plt.title('Animal Dataset Evolution Over Time')
    plt.grid(True)

    # Apply x-axis limits if provided
    if xmin is None:
        xmin = time[0]
    if xmax is None:
        xmax = time[-1]
    # print(f"Setting x-axis limits: xmin={xmin}, xmax={xmax}")
    plt.xlim(left=xmin, right=xmax)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    T = 10 * 24 * 60 * 60  # total time in seconds (10 days)

    # Define transition probabilities
    matrix = compute_balanced_transition_matrix(target_keep_ratio=0.95)
    for from_state, transitions in matrix.items():
        print(f"\nFrom {from_state}:")
        for to_state, prob in transitions.items():
            print(f"  → {to_state}: {prob:.4f}")

    transition_probs = flatten_transition_matrix(matrix)

    sim = DatasetEvolution(transition_probs, initial_state='KEEP', seed=123)

    # Simulate for T steps
    states = []
    for t in range(T):
        state = sim.step()
        states.append(state)

    # Map states to numeric values
    state_map = {'KEEP': 0, 'INSERT': 1, 'REMOVE': 2}
    numeric_states = [state_map[s] for s in states]

    # Plot using step
    plot_state_transitions(numeric_states, xmin=None, xmax=None, filename="animal_dataset_evolution.pdf")
    plot_state_transitions(numeric_states, xmin=None, xmax=100, filename="animal_dataset_evolution_zoomed.pdf")
