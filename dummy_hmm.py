import numpy as np

#HMM dummy implementation in numpy. It must be provided with the transition matrices A, B and the initial
#distribution pi.
#It is only use to generate data. It's not trainable.

class DummyHMM():
    # Not trainable HMM. User must provide tensors pi, A and B.
    def __init__(self, pi, A, B):
        self.pi = pi;
        self.A = A;
        self.B = B;
        self.n_states = A.shape[0]

    def get_log_likelihood(self, x):
        # Forward algorithm, scaled version to support long sequences
        T = len(x)
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)

        alpha[0] = self.pi * self.B[:, x[0]]
        c[0] = np.sum(alpha[0])
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = alpha[t - 1, :].dot(self.A) * self.B[:, x[t]];
            c[t] = np.sum(alpha[t])
            alpha[t] /= c[t]

        return np.sum(np.log2(c))

    def get_most_probable_hidden_sequence(self, x):
        # Viterbi algorithm
        T = len(x)
        delta = np.zeros((T, self.n_states))
        states = np.zeros((T, self.n_states), dtype=int)

        delta[0] = np.log2(self.pi) + np.log2(self.B[:, x[0]])

        for t in range(1, T):
            delta_t_1 = delta[t - 1].reshape(self.n_states, 1)
            delta[t] = np.max(delta_t_1 + np.log2(self.A), axis=0) + np.log2(self.B[:, x[t]])
            states[t] = np.argmax(delta_t_1 + np.log2(self.A), axis=0)

        # Backtrack

        sequence = np.zeros(T, dtype=int)
        sequence[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            sequence[t] = states[t + 1, sequence[t + 1]]
        return sequence;
