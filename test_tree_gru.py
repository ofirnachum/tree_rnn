import tree_gru
import tree_rnn

import numpy as np
import theano


def test():
    # very simple for now... just checks compilation and training step
    # TODO: better tests
    root = tree_rnn.BinaryNode(0)
    c1 = tree_rnn.BinaryNode(1)
    cc1 = tree_rnn.BinaryNode(2)
    ccc1 = tree_rnn.BinaryNode(3)
    cccc1 = tree_rnn.BinaryNode(5)
    cccc2 = tree_rnn.BinaryNode(6)
    ccc1.add_left(cccc1)
    ccc1.add_right(cccc2)
    cc1.add_left(ccc1)
    c1.add_right(cc1)
    root.add_left(c1)

    # check child sum
    model = tree_gru.ChildSumTreeGRU(10, 20, 30, 1)
    model.train_step(root, np.array([0]).astype(theano.config.floatX))

    # check n-ary
    model = tree_gru.NaryTreeGRU(10, 20, 30, 1)
    model.train_step(root, np.array([0]).astype(theano.config.floatX))
