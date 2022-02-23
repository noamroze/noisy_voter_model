import numpy as np
import matplotlib.pyplot as plt

def get_polygon_points(n):
    points = np.exp(1j * 2 * np.pi * np.arange(n) / n)
    x = np.real(points)
    y = np.imag(points)
    return x, y


def draw_polygon(n, color):
    x, y = get_polygon_points(n)
    plt.plot(x, y, 'o', color=color)
    for i in range(n):
        plt.plot([x[i], x[(i+1) % n]], [y[i], y[(i+1) % n]], '-', color=color)


def random_matching(n):
    assert n % 2 == 0
    vertices = list(range(n))
    np.random.shuffle(vertices)
    return list(zip(vertices[: n//2], vertices[n//2:]))


def plot_matching(matching, n, color):
    x, y = get_polygon_points(n)
    for i, j in matching:
        plt.plot([x[i], x[j]], [y[i], y[j]], '-', color=color)

def plot_set_with_boundary(n, edges, S, markersize):
    x, y = get_polygon_points(n)
    plt.plot(x[S], y[S], 'o', color='g', markersize=markersize, label="$S$")

    dS = []
    for v in S:
        for possible_neighbour in np.setdiff1d(np.arange(n), S):
            if (v, possible_neighbour) in edges or (possible_neighbour, v) in edges:
                dS.append(possible_neighbour)
    plt.plot(x[dS], y[dS], 'o', color='purple', markersize=markersize, label="$\partial S$")
    plt.legend()
    return dS


if __name__ == "__main__":
    n = 100
    matching = random_matching(n)
    draw_polygon(n, "blue")
    plot_matching(matching, n, 'red')
    S = np.random.choice(np.arange(n), n // 2, replace=False)
    # S = np.array(matching)[:n // 4, 0].tolist() + np.array(matching)[:n // 4, 1].tolist()
    base_edges = list(zip(np.arange(n-1), np.arange(1, n)))
    dS = plot_set_with_boundary(n, matching + base_edges, S, markersize=5)
    plt.show()
    print(f"|S|={len(S)}")
    print(f"|dS|={len(dS)}")
    print(f"h(S)={len(dS)/len(S)}")