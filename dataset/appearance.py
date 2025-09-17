import random


class ElementAppearance:
    """
        Simulates the appearance of an animal or class over time using a Markov process.
        States: 'APPEAR', 'NOSHOW'

    """


    def __init__(self, p_aa, p_an, p_na, p_nn, seed=None):
        if seed is not None:
            random.seed(seed)

        # Transition probabilities
        self.p_aa = p_aa  # APPEAR → APPEAR
        self.p_an = p_an  # APPEAR → NOSHOW
        self.p_na = p_na  # NOSHOW → APPEAR
        self.p_nn = p_nn  # NOSHOW → NOSHOW

        # Initial state
        self.state = 'NOSHOW'  # You can start with 'APPEAR' if preferred

    def step(self):
        if self.state == 'APPEAR':
            self.state = 'APPEAR' if random.random() < self.p_aa else 'NOSHOW'
        else:  # NOSHOW
            self.state = 'APPEAR' if random.random() < self.p_na else 'NOSHOW'
        return self.state

    def simulate(self, T):
        states = []
        for _ in range(T):
            states.append(self.step())
        return states

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parameters
    p_aa = 0.5
    p_an = 1 - p_aa
    p_nn = 0.9993742
    p_na = 1 - p_nn

    T = 24 * 60 * 60  # simulate for one day (in seconds)

    # Example usage
    simulator = ElementAppearance(p_aa=p_aa, p_an=p_an, p_na=p_na, p_nn=p_nn, seed=42)

    states = simulator.simulate(T)
    num_appear = sum(1 for state in states if state == 'APPEAR') / len(states) # Count the number of APPEAR states.
    # Plot the changes in states
    plt.step(range(T), states)
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.title('Animal appearance simulator - APPEAR probability: {:.4f}%'.format(num_appear * 100))
    plt.yticks(['APPEAR', 'NOSHOW'])
    plt.savefig('animal_appearance_simulator.pdf')
    plt.close()
