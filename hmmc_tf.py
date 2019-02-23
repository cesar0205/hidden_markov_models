import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
MVN = tf.contrib.distributions.MultivariateNormalDiag

def random_normalized(shape0, shape1):
    X = np.random.random((shape0, shape1))
    return X/np.sum(X, axis = 1, keepdims = True)


class HMMC():
    def __init__(self, M, K):
        self.M = M
        self.K = K
        self.sess = None

    def _build(self, X):
        M = self.M
        K = self.K
        D = self.D
        T = self.T
        N = self.N
        tf.reset_default_graph()

        x_input = tf.placeholder(dtype=tf.float32, shape=(None, D))

        # For the set params function
        self.pi_input = tf.placeholder(dtype=tf.float32, shape=(M,))
        self.a_input = tf.placeholder(dtype=tf.float32, shape=(M, M))
        self.r_input = tf.placeholder(dtype=tf.float32, shape=(M, K))
        self.mu_input = tf.placeholder(dtype=tf.float32, shape=(M, K, D))
        self.sigma_input = tf.placeholder(dtype=tf.float32, shape=(M, K, D))

        A = random_normalized(M, M)
        R = random_normalized(M, K)

        mu = np.zeros((M, K, D), dtype=np.float32)
        # sigma = np.zeros((M, K, D, D), dtype = np.float32)
        sigma = np.zeros((M, K, D)).astype(np.float32)
        pi = np.ones(M, dtype=np.float32) / M

        # Initialize mu and sigma with samples from the data
        for m in range(M):
            for k in range(K):
                rand_n = np.random.choice(N)
                rand_t = np.random.choice(T)

                mu[m, k] = X[rand_n][rand_t]
                sigma[m, k] = np.ones(D)

        # Create tensorflow variables

        A_tf = tf.Variable(A, dtype=tf.float32)
        R_tf = tf.Variable(R, dtype=tf.float32)
        pi_tf = tf.Variable(pi, dtype=tf.float32)

        A_softmax = tf.nn.softmax(A_tf)
        R_softmax = tf.nn.softmax(R_tf)
        pi_softmax = tf.nn.softmax(pi_tf)

        mu_tf = tf.Variable(mu, dtype=tf.float32)
        sigma_tf = tf.Variable(sigma, dtype=tf.float32)

        B_list = []
        for m in range(M):
            components = [];
            for k in range(K):
                comp = R_softmax[m, k] * MVN(mu_tf[m, k], sigma_tf[m, k]).prob(x_input)
                components.append(comp)
            B_m = tf.add_n(components)
            B_m = tf.clip_by_value(B_m, 0.0,
                                   1.0)  # If the sigma_tf takes small values, then MVN will give very large values and
            # B_m will also have large values. We need probabilities (0, 1)
            B_list.append(B_m)
        B_tf = tf.stack(B_list)

        def recursion(old_a_old_c, t):
            at = tf.matmul(old_a_old_c[0], A_softmax) * B_tf[:, t]
            c = tf.reduce_sum(at)
            at /= c;
            return (at, c)

        alpha_init = tf.reshape(pi_softmax * B_tf[:, 0], (1, M))
        self.alpha_init = alpha_init
        self.pi_softmax = pi_softmax

        alphas, cs = tf.scan(fn=recursion,
                             elems=np.arange(1, T),
                             initializer=(alpha_init, 1.0))
        self.cs = cs
        cost = -tf.reduce_sum(tf.log(cs) / tf.log(2.0))
        opt_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        self.set_pi = tf.assign(pi_tf, self.pi_input)
        self.set_a = tf.assign(A_tf, self.a_input)
        self.set_r = tf.assign(R_tf, self.r_input)
        self.set_mu = tf.assign(mu_tf, self.mu_input)
        self.set_sigma = tf.assign(sigma_tf, self.sigma_input)

        self.cost = cost
        self.opt_step = opt_step
        self.init = tf.global_variables_initializer()
        self.x_input = x_input;
        self.sigma_tf = sigma_tf;

        self.A_softmax = A_softmax
        self.R_softmax = R_softmax
        self.pi_softmax = pi_softmax

        self.mu_tf = mu_tf
        self.sigma_tf = sigma_tf
        self.B_tf = B_tf

    def get_individual_log_likelihood(self, x):
        return -self.sess.run(self.cost, feed_dict={self.x_input: x})

    def get_log_likelihood(self, X):
        lps = []
        for x in X:
            lps.append(self.get_individual_log_likelihood(x));
        return np.array(lps);

    def set_params(self, M, D, K, pi, A, R, mu, sigma):
        self.M = M
        self.D = D
        self.K = K
        self.sess.run([self.set_pi, self.set_a, self.set_r, self.set_mu, self.set_sigma],
                      feed_dict={self.pi_input: pi,
                                 self.a_input: A,
                                 self.r_input: R,
                                 self.mu_input: mu,
                                 self.sigma_input: sigma})

    def fit(self, X, n_epochs):
        self.T = X[0].shape[0]
        self.D = X[0].shape[1]
        self.N = len(X)
        if (self.sess is not None):
            self.sess.close();

        self._build(X)
        self.sess = tf.Session()
        self.sess.run(self.init)
        total_loss = []
        for epoch in range(n_epochs):
            losses = []
            for x in X:
                __, loss_res = self.sess.run([self.opt_step, self.cost], feed_dict={self.x_input: x})
                losses.append(loss_res)
            epoch_loss = np.sum(losses)
            total_loss.append(epoch_loss)
            print("Epoch:", epoch, "Train loss:", epoch_loss)

        plt.plot(total_loss)
        plt.title("Training costs")
        plt.xlabel("Epoch")
        plt.ylabel("Cost. log2(p)")
        plt.show()