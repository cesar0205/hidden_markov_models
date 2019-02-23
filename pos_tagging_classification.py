import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from dummy_hmm import DummyHMM

#Calculates accuracy score given a list of pos tags and a list of sentence ids.
def custom_accuracy_score(model, tag_list, sentence_list):
    for tags, sentence in zip(tag_list, sentence_list):
        predictions = [];
        labels = [];
        pred_tags = list(model.get_most_probable_hidden_sequence(sentence))
        predictions.extend(pred_tags)
        labels.extend(tags)
        return accuracy_score(labels, predictions)

#Calculates f1 score given a list of pos tags and a list of sentence ids.
def custom_f1_score(model, tag_list, sentence_list):
    for tags, sentence in zip(tag_list, sentence_list):
        predictions = [];
        labels = [];
        pred_tags = list(model.get_most_probable_hidden_sequence(sentence))
        predictions.extend(pred_tags)
        labels.extend(tags)
        return f1_score(labels, predictions, average=None).mean()

#Processes train and test sets. It transforms the sentences and tags into token ids.
def get_data():
    train_sentences = []
    train_tags = []
    word2idx = {'unk_': 0}
    #idx2word = ['unk_']
    tag2idx = {}
    #idx2tag = []
    word_counter = 1
    tag_counter = 0
    with open("./chunking/train.txt") as f:
        sentence = []
        tags = []
        for line in f:
            tokens = line.split()
            if(len(tokens) == 3):
                word, tag, __ = tokens;

                if(word not in word2idx):
                    word2idx[word] = word_counter;
                    word_counter += 1
                if(tag not in tag2idx):
                    tag2idx[tag] = tag_counter;
                    tag_counter += 1

                sentence.append(word2idx[word])
                tags.append(tag2idx[tag])
            else:
                train_sentences.append(sentence)
                train_tags.append(tags)
                sentence = []
                tags = []

    test_tags = []
    test_sentences = []
    with open("./chunking/test.txt") as f:
        sentence = []
        tags = []
        for line in f:
            tokens = line.split()
            if(len(tokens) == 3):
                word, tag, __ = tokens;

                if(word in word2idx):
                    wordidx = word2idx[word];
                else:
                    wordidx = word2idx['unk_']

                sentence.append(wordidx)
                tags.append(tag2idx[tag])
            else:
                test_sentences.append(sentence)
                test_tags.append(tags)
                sentence = []
                tags = []

    return train_tags, train_sentences, test_tags, test_sentences, word2idx

def main():
    #Get the train and test data sets
    train_tags, train_sentences, test_tags, test_sentences, word2idx = get_data()

    n_states = max([max(tags) for tags in train_tags]) + 1
    n_vocabulary = max([max(sentence) for sentence in train_sentences]) + 1

    #Compute the transition matrices as well as the initial distribution.
    smoothing = 0.01
    pi = np.zeros((n_states)) + smoothing
    A = np.zeros((n_states, n_states)) + smoothing
    B = np.zeros((n_states, n_vocabulary)) + smoothing

    for tags, sentence in zip(train_tags, train_sentences):
        pi[tags[0]] += 1
        for i in range(len(tags) - 1):
            A[tags[i], tags[i + 1]] += 1

        for i in range(len(sentence)):
            B[tags[i], sentence[i]] += 1

    #Normalization step
    A = A / np.sum(A, axis=1, keepdims=True)
    B = B / np.sum(B, axis=1, keepdims=True)
    pi = pi / np.sum(pi)

    #Instantiate the HMM model with the true parameters.
    model = DummyHMM(pi, A, B)

    print("Train accuracy:", custom_accuracy_score(model, train_tags, train_sentences))
    print("Test accuracy:", custom_accuracy_score(model, test_tags, test_sentences))
    print("Train f1 score:", custom_f1_score(model, train_tags, train_sentences))
    print("Test f1 score:", custom_f1_score(model, test_tags, test_sentences))


if __name__ == "__main__":
    main()