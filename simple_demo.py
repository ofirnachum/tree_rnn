import tree_rnn
import tree_lstm
import tree_gru
from test_tree_rnn import DummyBinaryRNN

import theano
import numpy as np
import random

NUM_EMB = 10
EMB_DIM = 10
HIDDEN_DIM = 10
OUTPUT_DIM = 1

NUM_ITER = 100000
MAX_DEPTH = 4


def get_trainable_model():
    # change this to the model of your choosing
    return tree_rnn.TreeRNN(NUM_EMB, EMB_DIM, HIDDEN_DIM, OUTPUT_DIM)


def get_groundtruth_model():
    model = DummyBinaryRNN(NUM_EMB, EMB_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.embeddings.set_value(
        model.embeddings.get_value() +
        np.random.randint(low=-1, high=2, size=(NUM_EMB, EMB_DIM)).astype(theano.config.floatX))
    return model


def get_groundtruth_label(root_node, model):
    label = model.predict(root_node)
    return label.reshape((-1,))


def get_random_binary_tree(min_depth, max_depth, num_vals, child_prob=0.7, _cur_depth=0):
    root = tree_rnn.BinaryNode(int(random.random() * num_vals))
    if max_depth <= 1:
        return root

    # left child
    if _cur_depth < min_depth or random.random() < child_prob:
        left_child = get_random_binary_tree(min_depth, max_depth - 1, num_vals,
                                            child_prob=child_prob,
                                            _cur_depth=_cur_depth + 1)
        root.add_left(left_child)

    # right child
    if _cur_depth < min_depth or random.random() < child_prob:
        right_child = get_random_binary_tree(min_depth, max_depth - 1, num_vals,
                                             child_prob=child_prob,
                                             _cur_depth=_cur_depth + 1)
        root.add_right(right_child)

    return root


def main(num_iter=NUM_ITER):
    groundtruth_model = get_groundtruth_model()
    trainable_model = get_trainable_model()

    losses = []
    for it in xrange(num_iter):
        tree = get_random_binary_tree(1, MAX_DEPTH, NUM_EMB, child_prob=0.7)
        label = get_groundtruth_label(tree, groundtruth_model)

        loss, pred_y = trainable_model.train_step(tree, label)
        losses.append(loss)

        if it % 1000 == 0:
            print 'iter', it, ':', np.mean(losses), pred_y, label
            losses = []


if __name__ == '__main__':
    main()
