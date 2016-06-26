import tree_rnn

import theano
from theano import tensor as T
import numpy as np
from numpy.testing import assert_array_almost_equal


class DummyTreeRNN(tree_rnn.TreeRNN):

    def create_recursive_unit(self):
        def unit(parent_x, child_h, child_exists):  # assumes emb_dim == hidden_dim
            return parent_x + T.prod((child_h - 1) * child_exists.dimshuffle(0, 'x') + 1,
                                     axis=0)
        return unit

    def create_leaf_unit(self):
        def unit(leaf_x):  # assumes emb_dim == hidden_dim
            return leaf_x
        return unit


class DummyBinaryRNN(tree_rnn.TreeRNN):

    def create_recursive_unit(self):
        def unit(parent_x, child_h, child_exists):  # assumes emb_dim == hidden_dim
            return (parent_x + child_exists[0] * child_h[0] +
                    child_exists[1] * child_h[1] ** 2)
        return unit

    def create_leaf_unit(self):
        def unit(leaf_x):  # assumes emb_dim == hidden_dim
            return leaf_x
        return unit


def test_tree_rnn():
    model = DummyTreeRNN(8, 2, 2, 1, degree=2)
    emb = model.embeddings.get_value()

    root = tree_rnn.Node(3)
    c1 = tree_rnn.Node(1)
    c2 = tree_rnn.Node(2)
    root.add_children([c1, c2])

    root_emb = model.evaluate(root)
    expected = emb[3] + emb[1] * emb[2]
    assert_array_almost_equal(expected, root_emb)

    cc1 = tree_rnn.Node(5)
    cc2 = tree_rnn.Node(2)
    c2.add_children([cc1, cc2])

    root_emb = model.evaluate(root)
    expected = emb[3] + (emb[2] + emb[5] * emb[2]) * emb[1]
    assert_array_almost_equal(expected, root_emb)

    ccc1 = tree_rnn.Node(5)
    ccc2 = tree_rnn.Node(4)
    cc1.add_children([ccc1, ccc2])

    root_emb = model.evaluate(root)
    expected = emb[3] + (emb[2] + (emb[5] + emb[5] * emb[4]) * emb[2]) * emb[1]
    assert_array_almost_equal(expected, root_emb)

    # check step works without error
    model.train_step(root, np.array([0]).astype(theano.config.floatX))

    # degree > 2
    model = DummyTreeRNN(10, 2, 2, 1, degree=3)
    emb = model.embeddings.get_value()

    root = tree_rnn.Node(0)
    c1 = tree_rnn.Node(1)
    c2 = tree_rnn.Node(2)
    c3 = tree_rnn.Node(3)
    root.add_children([c1, c2, c3])

    cc1 = tree_rnn.Node(1)
    cc2 = tree_rnn.Node(2)
    cc3 = tree_rnn.Node(3)
    cc4 = tree_rnn.Node(4)
    cc5 = tree_rnn.Node(5)
    cc6 = tree_rnn.Node(6)
    cc7 = tree_rnn.Node(7)
    cc8 = tree_rnn.Node(8)
    cc9 = tree_rnn.Node(9)

    c1.add_children([cc1, cc2, cc3])
    c2.add_children([cc4, cc5, cc6])
    c3.add_children([cc7, cc8, cc9])

    root_emb = model.evaluate(root)
    expected = \
        emb[0] + ((emb[1] + emb[1] * emb[2] * emb[3]) *
                  (emb[2] + emb[4] * emb[5] * emb[6]) *
                  (emb[3] + emb[7] * emb[8] * emb[9]))
    assert_array_almost_equal(expected, root_emb)

    # check step works without error
    model.train_step(root, np.array([0]).astype(theano.config.floatX))


def test_tree_rnn_var_degree():
    model = DummyBinaryRNN(10, 2, 2, 1, degree=2)
    emb = model.embeddings.get_value()

    root = tree_rnn.BinaryNode(0)
    c1 = tree_rnn.BinaryNode(1)
    cc1 = tree_rnn.BinaryNode(2)
    ccc1 = tree_rnn.BinaryNode(3)
    cc1.add_left(ccc1)
    c1.add_right(cc1)
    root.add_left(c1)

    root_emb = model.evaluate(root)
    expected = emb[0] + (emb[1] + (emb[2] + emb[3]) ** 2)
    assert_array_almost_equal(expected, root_emb)

    cccc1 = tree_rnn.BinaryNode(5)
    cccc2 = tree_rnn.BinaryNode(6)
    ccc1.add_left(cccc1)
    ccc1.add_right(cccc2)

    root_emb = model.evaluate(root)
    expected = emb[0] + (emb[1] + (emb[2] + (emb[3] + emb[5] + emb[6] ** 2)) ** 2)
    assert_array_almost_equal(expected, root_emb)

    # check step works without error
    model.train_step(root, np.array([0]).astype(theano.config.floatX))


def test_irregular_tree():
    model = DummyTreeRNN(8, 2, 2, 1, degree=4, irregular_tree=True)
    emb = model.embeddings.get_value()

    root = tree_rnn.Node(3)
    c1 = tree_rnn.Node(1)
    c2 = tree_rnn.Node(2)
    c3 = tree_rnn.Node(3)
    c4 = tree_rnn.Node(4)
    c5 = tree_rnn.Node(5)
    c6 = tree_rnn.Node(6)
    root.add_children([c1, c2, c3, c4])
    c1.add_children([c5])
    c5.add_children([c6])

    root_emb = model.evaluate(root)
    expected = emb[3] + emb[2] * emb[3] * emb[4] * (emb[1] + emb[5] + emb[6])
    assert_array_almost_equal(expected, root_emb)
