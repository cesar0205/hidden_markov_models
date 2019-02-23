import tensorflow as tf
import numpy as np
from nltk import word_tokenize, pos_tag
from sklearn.utils import shuffle
from hmmd_tf import HMMD

#Example of how to use generative models such as hidden markov models for classification


def get_data():
    # Because the vocabulary will be too large compared to the relatively small data sets, we are likely to overfit.
    # Then instead of extracting the vocabulary we will extract the pos_tags.
    X = []
    y = []
    word2idx = {}
    idx2word = []

    count = 0;
    for file_name, label in zip(("./edgar_allan_poe.txt", "./robert_frost.txt"), (0, 1)):
        with open(file_name) as f:
            for line in f:
                tags = pos_tag(word_tokenize(line.lower()))
                # print(tags)
                sentence = []
                if len(tags) > 1:
                    for word, tag in tags:
                        if tag not in word2idx:
                            word2idx[tag] = count;
                            idx2word.append(tag)
                            count += 1;
                        sentence.append(word2idx[tag])
                    X.append(sentence);
                    y.append(label);

    return X, y, word2idx, idx2word


from sklearn.metrics import accuracy_score, f1_score


class HMMClassifier():
    # HMM classifier for discrete data.
    # The HMMClassifier will create a discrete HMM model for each class in the training set.
    def __init__(self, n_states, n_outputs, learning_rate=0.2):
        self.learning_rate = learning_rate;
        self.n_states = n_states;
        self.n_outputs = n_outputs;

    def fit(self, X, y, n_epochs):
        self.classes = np.unique(y);
        self.models = []
        # Fit a different model to each one of our classes
        for class_ in self.classes:
            X_class = [x for x, y in zip(X, y) if y == class_];
            model = HMMD(self.n_states, self.n_outputs, learning_rate=self.learning_rate);
            model.fit(X_class, n_epochs);
            self.models.append(model);

        self.priors = []
        for class_ in self.classes:
            n_label = len([label for label in y if label == class_])
            prior = n_label / len(y)
            self.priors.append(prior);

    def predict(self, X):
        # Calculate the unnormalized posteriors, that will be used as discriminators.
        un_posteriors = np.zeros((len(X), len(self.classes)))
        for class_ in self.classes:
            un_posteriors[:, class_] = self.models[class_].get_multiple_log_likelihood(X) + np.log2(self.priors[class_])

        return np.argmax(un_posteriors, axis=1)

    def f1_score(self, X, y_true):
        return f1_score(y_true, self.predict(X))

    def accuracy_score(self, X, y_true):
        return accuracy_score(y_true, self.predict(X))

def main():
    # Loads poems from two authors and processes the data
    X, y, word2idx, idx2word = get_data()
    # Split to train, test sets

    n_train = int(0.8 * len(X))
    X, y = shuffle(X, y)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    # Size of output vocabulary
    n_vocabulary = max([max(sentence) for sentence in X]) + 1

    # Train the discrete HMM classifier
    model = HMMClassifier(n_states=10, n_outputs=n_vocabulary, learning_rate=0.1)
    model.fit(X_train, y_train, 50)

    # Print scores
    print("Train accuracy:", model.accuracy_score(X_train, y_train))
    print("Test accuracy:", model.accuracy_score(X_test, y_test))
    print("Train f1 score:", model.f1_score(X_train, y_train))
    print("Test f1 score:", model.f1_score(X_test, y_test))

if __name__ == "__main__":
    main();