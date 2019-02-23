# Hidden Markov Models

## a) Continuous HMM for Gaussian Mixuture

Study case of a Hidden Markov Model for continuous data. The model is created in Tensorflow

Data is generated using a trasition matrix:

    A = np.array([
        [0.9, 0.025, 0.025, 0.025, 0.025],
        [0.025, 0.9, 0.025, 0.025, 0.025],
        [0.025, 0.025, 0.9, 0.025, 0.025],
        [0.025, 0.025, 0.025, 0.9, 0.025],
        [0.025, 0.025, 0.025, 0.025, 0.9],
    ])

The mixture components:

    R = np.ones((M, K)) / K

The mu locations:

    mu = np.array([
        [[0, 0], [1, 1], [2, 2]],
        [[5, 5], [6, 6], [7, 7]],
        [[10, 10], [11, 11], [12, 12]],
        [[15, 15], [16, 16], [17, 17]],
        [[20, 20], [21, 21], [22, 22]],
    ])
    
And the covariance matrices:

    sigma[m, k] = np.identity(D)


The tensorflow model follows the scaled version of Forward Algorithm for HMM to support long sequences.

To calculate B we just sum over all k states of 

    R[j, k]*likelihood(x(t) | mu[j, k], sigma[j, k])
 
The backward algorithm and the Baum-Welch are not used as the loss can be calculated directy with the negative of the log probability of a sequence.

To calculate the alpha matrix we use the tensorflow scan operator to create a recurrent function.

    def recursion(old_a_old_c, t):
         at = tf.matmul(old_a_old_c[0], A_softmax) * B_tf[:, t]
         c = tf.reduce_sum(at)
         at/=c;
         return (at, c)

The cost of a sample is calculated as the log of that sequence probability. We use the log 
because as the sequence length gets larger the probability becomes close to zero creating an underflow problem.

    cost = -tf.reduce_sum(tf.log(cs)/tf.log(2.0))
    
We can then use gradient descent or another optimization algorithm to update the matrices.

    opt_step = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)


After the test data is generated, the HMM is trained and the training costs are plotted.

![Train cost](hdmm_costs.png)

## b) Discrete HMM for text classification

Study case where generative models such as HMMs can be used in conjunction with the Bayes Rule to create a discriminator.

The code for a disrete HMM is similar to the continuous case. However, we can model the matrix B (observations matrix) directly and normalize its entries.

    B = tf.Variable(np.random.rand(self.n_states, self.n_outputs), dtype=tf.float32)
    B_norm = tf.nn.softmax(B)

We have two files with writings from Edgar Allan Poe and from Robert Fros. The objective is to classify a new sentence to either of the two authors.

If we use the total vocabulary extracted from these files, then the number of outputs will be too large compared to the relatively small data sets and we are likely to overfit to the train set.
Instead, we will extract the pos_tags using nltk pos_tag function. The output size will be 41.

For each class (author) a different model will be trained and the prior calculated as well.

Finally we can use bayes rule to compute the unnormalized posterior that we can utilize as discriminator to decide the class that corresponds to a given sentence.
    
    def predict(self, X):
            # Calculate the unnormalized posteriors, that will be used as discriminators.
            un_posteriors = np.zeros((len(X), len(self.classes)))
            for class_ in self.classes:
                un_posteriors[:, class_] = self.models[class_].get_multiple_log_likelihood(X) + np.log2(self.priors[class_])
    
            return np.argmax(un_posteriors, axis=1)

As classes in the data are skewed we calculate the F1 score in addition to the accuracy score in order to have a better comparitive for the performances on the train and test sets.

    ...
    Epoch: 46 Train loss: 23202.0368384
    Epoch: 47 Train loss: 23163.7684468
    Epoch: 48 Train loss: 23126.3661596
    Epoch: 49 Train loss: 23089.7905619
    
    
    Train accuracy: 0.735925710969
    Test accuracy: 0.751740139211
    Train f1 score: 0.808097849009
    Test f1 score: 0.818950930626
    
This is an acceptable result. We must recall that generally generative models (as in this case) perform worse than discriminative models for classification tasks even though
the former are more expressive in regards to explain the cause of observations. 

## c) Discrete HMM for pos tagging

The task is to assign a word in a sentence its corresponding pos_tag. As in the previous examples we also create hidden variables to create a generative model.
The hidden variables represent the different pos-tags in our data and the observable variables are the vocabulary extracted from the sentences.

However, in this particular case we can compute directy the transition probabilities of our hidden states and observable states just by counting. This way we don't have to use the EM
algorithm to maximize the the likelihood of the data.

    for tags, sentence in zip(train_tags, train_sentences):
        pi[tags[0]] += 1
        for i in range(len(tags) - 1):
            A[tags[i], tags[i + 1]] += 1

        for i in range(len(sentence)):
            B[tags[i], sentence[i]] += 1 
            
After the counting step we have to normalize the matrices.

    #Normalization step
    A = A / np.sum(A, axis=1, keepdims=True)
    B = B / np.sum(B, axis=1, keepdims=True)
    pi = pi / np.sum(pi)
    
When we want to get the pos tags related to a observable sequence it is necessary to run the viterby algorithm to get
the most probable hidden sequence (see the code for implementation details).

Finally we can print both the accuracy and f1 scores.

    Train accuracy: 1.0
    Test accuracy: 0.928571428571
    Train f1 score: 1.0
    Test f1 score: 0.841666666667

This is a good result. However we must compare it with a logistic regression model and more sophisticated sequence classifiers such
RNN to have a benchmark for reference.


