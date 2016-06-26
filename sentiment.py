import data_utils
import tree_rnn
import tree_lstm
import tree_gru

import numpy as np
import theano
from theano import tensor as T
import random
import pickle
import os

DIR = '../treelstm/data/sst'
GLOVE_DIR = '../treelstm/data'  # should include .npy files of glove vecs and words
FINE_GRAINED = False
DEPENDENCY = False
SEED = 88

NUM_EPOCHS = 30
LEARNING_RATE = 0.01

EMB_DIM = 300
HIDDEN_DIM = 100


class SentimentModel(tree_lstm.ChildSumTreeLSTM):
    def train_step_inner(self, x, tree, y, y_exists):
        self._check_input(x, tree)
        return self._train(x, tree[:, :-1], y, y_exists)

    def train_step(self, root_node, label):
        x, tree, labels, labels_exist = \
            tree_rnn.gen_nn_inputs(root_node, max_degree=self.degree,
                                   only_leaves_have_vals=False,
                                   with_labels=True)
        y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
        y[np.arange(len(labels)), labels.astype('int32')] = 1
        loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
        return loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        return T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)


def get_model(num_emb, output_dim, max_degree):
    return SentimentModel(
        num_emb, EMB_DIM, HIDDEN_DIM, output_dim,
        degree=max_degree, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,
        labels_on_nonroot_nodes=True,
        irregular_tree=DEPENDENCY)

def train():
    vocab, data = data_utils.read_sentiment_dataset(DIR, FINE_GRAINED, DEPENDENCY)

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    max_degree = data['max_degree']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    print 'test', len(test_set)
    print 'max degree', max_degree

    num_emb = vocab.size()
    num_labels = 5 if FINE_GRAINED else 3
    for key, dataset in data.items():
        if key == 'max_degree':
            continue
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)
    print 'num emb', num_emb
    print 'num labels', num_labels

    random.seed(SEED)
    np.random.seed(SEED)
    model = get_model(num_emb, num_labels, max_degree)

    # initialize model embeddings to glove
    embeddings = model.embeddings.get_value()
    glove_vecs = np.load(os.path.join(GLOVE_DIR, 'glove.npy'))
    glove_words = np.load(os.path.join(GLOVE_DIR, 'words.npy'))
    glove_word2idx = dict((word, i) for i, word in enumerate(glove_words))
    for i, word in enumerate(vocab.words):
        if word in glove_word2idx:
            embeddings[i] = glove_vecs[glove_word2idx[word]]
    glove_vecs, glove_words, glove_word2idx = [], [], []
    model.embeddings.set_value(embeddings)

    for epoch in xrange(NUM_EPOCHS):
        print 'epoch', epoch
        avg_loss = train_dataset(model, train_set)
        print 'avg loss', avg_loss
        dev_score = evaluate_dataset(model, dev_set)
        print 'dev score', dev_score

    print 'finished training'
    test_score = evaluate_dataset(model, test_set)
    print 'test score', test_score


def train_dataset(model, data):
    losses = []
    avg_loss = 0.0
    total_data = len(data)
    for i, (tree, _) in enumerate(data):
        loss, pred_y = model.train_step(tree, None)  # labels will be determined by model
        losses.append(loss)
        avg_loss = avg_loss * (len(losses) - 1) / len(losses) + loss / len(losses)
        print 'avg loss %.2f at example %d of %d\r' % (avg_loss, i, total_data),
    return np.mean(losses)


def evaluate_dataset(model, data):
    num_correct = 0
    for tree, label in data:
        pred_y = model.predict(tree)[-1]  # root pred is final row
        num_correct += (label == np.argmax(pred_y))

    return float(num_correct) / len(data)


if __name__ == '__main__':
    train()
