import numpy as np
import matplotlib.pyplot as plt

def get_small_params():
    M = 1
    D = 1
    K = 1
    pi = np.array([1])
    A = np.array([[1]])
    R = np.array([[1]])
    mu = np.array([[[0]]])
    sigma = np.array([[[[1]]]])
    return M, D, K, pi, A, R, mu, sigma;

def get_params():

    M = 5  # States
    D = 2
    K = 3  # Components

    pi = np.array([1, 0, 0, 0, 0])  # initial state distribution

    A = np.array([
        [0.9, 0.025, 0.025, 0.025, 0.025],
        [0.025, 0.9, 0.025, 0.025, 0.025],
        [0.025, 0.025, 0.9, 0.025, 0.025],
        [0.025, 0.025, 0.025, 0.9, 0.025],
        [0.025, 0.025, 0.025, 0.025, 0.9],
    ])  # state transition matrix - likes to stay where it is

    R = np.ones((M, K)) / K  # mixture proportions

    mu = np.array([
        [[0, 0], [1, 1], [2, 2]],
        [[5, 5], [6, 6], [7, 7]],
        [[10, 10], [11, 11], [12, 12]],
        [[15, 15], [16, 16], [17, 17]],
        [[20, 20], [21, 21], [22, 22]],
    ])  # M x K x D

    sigma = np.zeros((M, K, D, D))
    for m in range(M):
        for k in range(K):
            sigma[m, k] = np.identity(D)

    return M, D, K, pi, A, R, mu, sigma




def generate_fake_data(N, T, init = get_params):
    '''
        N. Number of samples
        T. Timesteps
    '''

    M, D, K, pi, A, R, mu, sigma = init()

    data = []
    X = np.zeros((T, D))
    for i in range(N):

        s = 0  # As initial distribution concentrates all the weight on s_0
        r0 = R[s]  # Mixture proportions for state s
        c0 = np.random.choice(K, p=r0)  # Sample mixture with probability rs
        x0 = np.random.multivariate_normal(mu[s][c0], sigma[s][c0])
        X[0] = x0

        for t in range(1, T):
            s = np.random.choice(M, p=A[s])
            rt = R[s]  # Mixture proportions for state s
            ct = np.random.choice(K, p=rt)  # Sample mixture with probability rs
            xt = np.random.multivariate_normal(mu[s][ct], sigma[s][ct])
            X[t] = xt
        data.append(X)
    return np.array(data)


if __name__ == '__main__':
    X = generate_fake_data(1, 500, get_small_params)[0]
    print(X)

    plt.plot(X[:, 0], ".")
    plt.show()
