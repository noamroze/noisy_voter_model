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

    def random_neighbours(self, vote_indices):
        raise NotImplementedError

    def vote(self):
        new_votes = np.empty(self.n)
        vote_indices = np.random.random(self.n) > self.epsilon
        new_votes[~vote_indices] = np.random.randint(0, 2, (~vote_indices).sum())

        random_neighbours = self.random_neighbours(np.nonzero(vote_indices)[0])
        new_votes[vote_indices] = self.votes[random_neighbours]
        self.votes = new_votes

    def iter(self, T):
        for _ in range(T):
            yield self.votes
            self.vote()


class CompleteGraphNoisyVoter(NoisyVoter):
    def random_neighbours(self, vote_indices):
        return  (vote_indices + np.random.randint(0, self.n, len(vote_indices))) % self.n


class CycleNoisyVoter(NoisyVoter):
    def random_neighbours(self, vote_indices):
        return (vote_indices + (-1)**np.random.randint(0, 2, len(vote_indices))) % self.n


def get_votes_matrix(model, T):
    return np.concatenate([votes.reshape(-1, 1) for votes in model.iter(T)], axis=1)


def plot_number_of_ones_over_time(model, T):
    n_ones = get_votes_matrix(model, T).sum(axis=0)
    plt.plot(n_ones / model.n)
    plt.show()


def plot_heatmap(model, T):
    mat = get_votes_matrix(model, T)
    fig, ax = plt.subplots()
    im = ax.imshow(mat, interpolation='none')
    plt.show()

if __name__ == '__main__':
    model = CycleNoisyVoter(n=100, epsilon=0.0000)
    T = 1000

    plot_heatmap(model, T)
    plt.show()
    plot_number_of_ones_over_time(model, T)
    plt.show()
