import numpy as np
import matplotlib.pyplot as plt


class NoisyVoter:
    def __init__(self, n, epsilon, init=None):
        self.n = n
        self.epsilon = epsilon
        if init is None:
            self.votes = np.random.randint(0, 2, n)
        else:
            assert len(init) == n
            assert sorted(np.unique(init)) == [0, 1]
            self.votes = init

    def random_neighbour(self, v):
        raise NotImplementedError

    def vote(self):
        v = np.random.randint(0, self.n)
        if np.random.random() > self.epsilon:
            neighbour = self.random_neighbour(v)
            new_vote = self.votes[neighbour]
        else:
            new_vote = np.random.randint(0, 2)
        self.votes[v] = new_vote

    def iter(self, T):
        init_votes = self.votes
        for _ in range(T):
            yield self.votes
            self.vote()
        # reset votes
        self.votes = init_votes


class CompleteGraphNoisyVoter(NoisyVoter):
    def random_neighbour(self, v):
        return (v + np.random.randint(1, self.n)) % self.n


class CycleNoisyVoter(NoisyVoter):
    def random_neighbour(self, v):
        return (v + np.random.choice([-1, 1])) % self.n


def get_votes_matrix(model, T):
    res = []
    for votes in model.iter(T):
        res.append(votes.copy().reshape(-1, 1))
    return np.concatenate(res, axis=1)


def plot_number_of_ones_over_time(model, T):
    n_ones = np.empty(T)
    for i, votes in enumerate(model.iter(T)):
        n_ones[i] = votes.sum()
    subsample = np.arange(0, T, T // 100)

    plt.plot(n_ones[subsample] / model.n)
    plt.show()


def plot_heatmap(model, T):
    mat = get_votes_matrix(model, T)
    fig, ax = plt.subplots()
    subsample = np.arange(0, T, T // 100)
    im = ax.imshow(mat[:, subsample], interpolation='none')
    plt.show()

def plot_vertex_votes(model, T, v):
    subsample = np.arange(0, T, T // 100)
    votes = np.array([votes[v].copy() for votes in model.iter(T)])
    plt.plot(votes[subsample])
    plt.show()

if __name__ == '__main__':
    n = 1000
    model = CycleNoisyVoter(n, epsilon=0)
    T = 100000

    plot_heatmap(model, T)
    plot_number_of_ones_over_time(model, T)
    plot_vertex_votes(model, T, np.random.randint(0, n))

