__doc__ = """Implementation of Tree GRUs, and adaptation of GRU RNNs to trees."""

import tree_rnn

import theano
from theano import tensor as T


def _softmax(inp, exists, add_one=False):
    """Equivalent to T.nnet.softmax, but allowing for ignoring some columns.

    Also works on rows rather than columns.

    """
    inp = inp * exists.dimshuffle(0, 'x')
    e_inp = T.exp(inp - inp.max(axis=0, keepdims=True)) * exists.dimshuffle(0, 'x')
    if add_one:
        return e_inp / (1 + e_inp.sum(axis=0, keepdims=True))
    else:
        return e_inp / e_inp.sum(axis=0, keepdims=True)


class ChildSumTreeGRU(tree_rnn.TreeRNN):
    def create_recursive_unit(self):
        self.W_z = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_z = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.W_r = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_r = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.W_h = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_h = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.params.extend([
            self.W_z, self.U_z,
            self.W_r, self.U_r,
            self.W_h, self.U_h])

        def unit(parent_x, child_h, child_exists):
            z = _softmax(
                (T.dot(self.W_z, parent_x).dimshuffle('x', 0) +
                 T.dot(child_h, self.U_z.T)),
                child_exists, add_one=True)
            r = _softmax(
                (T.dot(self.W_r, parent_x).dimshuffle('x', 0) +
                 T.dot(child_h, self.U_r.T)),
                child_exists, add_one=False)
            h_hat = T.tanh(T.dot(self.W_h, parent_x) +
                           T.dot(self.U_h, T.sum(r * child_h, axis=0)))
            h = (1 - T.sum(z, axis=0)) * h_hat + T.sum(z * child_h, axis=0)
            return h

        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            return self.recursive_unit(
                leaf_x,
                dummy,
                1 + dummy.sum(axis=1))
        return unit


class NaryTreeGRU(ChildSumTreeGRU):
    # TODO: try a more analgous to LSTM degree ** 2 version

    def create_recursive_unit(self):
        self.W_z = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_z = theano.shared(self.init_matrix(
            [self.degree, self.hidden_dim, self.hidden_dim]))
        self.W_r = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_r = theano.shared(self.init_matrix(
            [self.degree, self.hidden_dim, self.hidden_dim]))
        self.W_h = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_h = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.params.extend([
            self.W_z, self.U_z,
            self.W_r, self.U_r,
            self.W_h, self.U_h])

        def unit(parent_x, child_h, child_exists):
            (pre_z, pre_r), _ = theano.map(
                fn=lambda Uz, Ur, h: (T.dot(Uz, h), T.dot(Ur, h)),
                sequences=[self.U_z, self.U_r, child_h])

            z = _softmax(
                T.dot(self.W_z, parent_x).dimshuffle('x', 0) + pre_z,
                child_exists, add_one=True)
            r = _softmax(
                T.dot(self.W_r, parent_x).dimshuffle('x', 0) + pre_r,
                child_exists, add_one=False)
            h_hat = T.tanh(T.dot(self.W_h, parent_x) +
                           T.dot(self.U_h, T.sum(r * child_h, axis=0)))
            h = (1 - T.sum(z, axis=0)) * h_hat + T.sum(z * child_h, axis=0)
            return h

        return unit
