import tensorflow as tf
import numpy as np

def normalized_random(dim1, dim2):
    A = np.random.rand(dim1, dim2);
    return A / np.sum(A, axis=1, keepdims=True)


class HMM():
    def __init__(self, n_states=2, n_outputs = None, learning_rate = 0.01):
        self.learning_rate = learning_rate;
        self.n_outputs = n_outputs;
        self.n_states = n_states;
        self.sess = None;

    def _build(self):

        tf.reset_default_graph()

        x_input = tf.placeholder(dtype=tf.int32, shape=(None,))
        a_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.n_states, self.n_states))
        b_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.n_states, self.n_outputs))
        pi_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.n_states))

        A = tf.Variable(np.random.rand(self.n_states, self.n_states), dtype=tf.float32)
        B = tf.Variable(np.random.rand(self.n_states, self.n_outputs), dtype=tf.float32)
        pi = tf.Variable(np.random.rand(self.n_states), dtype=tf.float32)

        A_norm = tf.nn.softmax(A)
        B_norm = tf.nn.softmax(B)
        pi_norm = tf.nn.softmax(pi)

        def recursion(old_a_old_c, x):
            at = tf.matmul(old_a_old_c[0], A_norm) * B_norm[:, x]
            c = tf.reduce_sum(at)
            at /= c;
            return (at, c)

        alpha_init = tf.reshape(pi * B_norm[:, x_input[0]], (1, self.n_states))
        alphas, cs = tf.scan(fn=recursion,
                             elems=x_input[1:],
                             initializer=(alpha_init, 1.0))

        loss = -tf.reduce_sum(
            tf.log(cs) / tf.log(2.0))  # If we keep the positive sign we want to maximize the log_probability
        # If wee add the negative sign we want to minimize it.

        opt_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        self.assign_A = tf.assign(A, a_placeholder);  # In case I want to test with a custom distribution.
        self.assign_B = tf.assign(B, b_placeholder);
        self.assign_pi = tf.assign(pi, pi_placeholder);

        init = tf.global_variables_initializer();

        self.x_input = x_input;
        self.a_placeholder = a_placeholder;
        self.b_placeholder = b_placeholder;
        self.pi_placeholder = pi_placeholder;
        self.init = init;
        self.loss = loss;
        self.opt_step = opt_step;

    def fit(self, X, n_epochs=10):

        if (self.sess is not None):
            self.sess.close()

        if (self.n_outputs is None):
            self.n_outputs = max([max(x) for x in X]) + 1

        self._build();

        self.sess = tf.Session();
        self.sess.run(self.init);

        epoch_losses = []
        for epoch in range(n_epochs):
            total_loss = 0
            for x in X:
                __, loss = self.sess.run([self.opt_step, self.loss], feed_dict={self.x_input: x})
                total_loss += loss

            epoch_losses.append(total_loss)
            print("Epoch:", epoch, "Train loss:", total_loss)

    def get_loss(self, X):
        total_loss = 0
        for x in X:
            loss = self.sess.run(self.loss, feed_dict={self.x_input: x})
            total_loss += loss
        return total_loss;

    def set_weights(self, pi, A, B):
        self.sess.run([self.assign_A, self.assign_B, self.assign_pi], feed_dict={self.a_placeholder: A,
                                                                                 self.b_placeholder: B,
                                                                                 self.pi_placeholder: pi})

    def get_log_likelihood(self, x):
        neg_log_prob = self.sess.run(self.loss, feed_dict={self.x_input: x})
        return -neg_log_prob;

    def get_multiple_log_likelihood(self, X):
        p = [];
        for x in X:
            p.append(self.get_log_likelihood(x));
        return np.array(p);


    #Agregar score o accuracy