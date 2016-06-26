__doc__ = """Implementation of Tree LSTMs described in http://arxiv.org/abs/1503.00075"""

import tree_rnn

import theano
from theano import tensor as T


class ChildSumTreeLSTM(tree_rnn.TreeRNN):
    def create_recursive_unit(self):
        self.W_i = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_i = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_i = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_f = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_f = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_f = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_o = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_o = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_o = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_u = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_u = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_u = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_u, self.U_u, self.b_u])

        def unit(parent_x, child_h, child_c, child_exists):
            h_tilde = T.sum(child_h, axis=0)
            i = T.nnet.sigmoid(T.dot(self.W_i, parent_x) + T.dot(self.U_i, h_tilde) + self.b_i)
            o = T.nnet.sigmoid(T.dot(self.W_o, parent_x) + T.dot(self.U_o, h_tilde) + self.b_o)
            u = T.tanh(T.dot(self.W_u, parent_x) + T.dot(self.U_u, h_tilde) + self.b_u)

            f = (T.nnet.sigmoid(
                    T.dot(self.W_f, parent_x).dimshuffle('x', 0) +
                    T.dot(child_h, self.U_f.T) +
                    self.b_f.dimshuffle('x', 0)) *
                 child_exists.dimshuffle(0, 'x'))

            c = i * u + T.sum(f * child_c, axis=0)
            h = o * T.tanh(c)
            return h, c

        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            return self.recursive_unit(
                leaf_x,
                dummy,
                dummy,
                dummy.sum(axis=1))
        return unit

    def compute_tree(self, emb_x, tree):
        self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        if self.irregular_tree:
            init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
            init_node_c = T.concatenate([leaf_c, leaf_c], axis=0)
        else:
            init_node_h = leaf_h
            init_node_c = leaf_c

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, node_c, last_h):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h, parent_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c,
                                    parent_c.reshape([1, self.hidden_dim])])
            return node_h[1:], node_c[1:], parent_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, init_node_c, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)


class NaryTreeLSTM(ChildSumTreeLSTM):
    # we inherit from ChildSumTreeLSTM to re-use the compute_tree method

    def create_recursive_unit(self):
        self.W_i = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_i = theano.shared(self.init_matrix(
            [self.degree, self.hidden_dim, self.hidden_dim]))
        self.b_i = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_f = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_f = theano.shared(self.init_matrix(
            [self.degree, self.degree, self.hidden_dim, self.hidden_dim]))
        self.b_f = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_o = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_o = theano.shared(self.init_matrix(
            [self.degree, self.hidden_dim, self.hidden_dim]))
        self.b_o = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_u = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_u = theano.shared(self.init_matrix(
            [self.degree, self.hidden_dim, self.hidden_dim]))
        self.b_u = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_u, self.U_u, self.b_u])

        def unit(parent_x, child_h, child_c, child_exists):
            (h_i, h_o, h_u), _ = theano.map(
                fn=lambda Ui, Uo, Uu, h, exists:
                    (exists * T.dot(Ui, h), exists * T.dot(Uo, h), exists * T.dot(Uu, h)),
                sequences=[self.U_i, self.U_o, self.U_u, child_h, child_exists])

            i = T.nnet.sigmoid(T.dot(self.W_i, parent_x) + h_i.sum(axis=0) + self.b_i)
            o = T.nnet.sigmoid(T.dot(self.W_o, parent_x) + h_o.sum(axis=0) + self.b_o)
            u = T.tanh(T.dot(self.W_u, parent_x) + h_u.sum(axis=0) + self.b_u)

            def _sub_f(U):
                sub_h_f, _ = theano.map(
                    fn=lambda sub_U, h, exists: exists * T.dot(sub_U, h),
                    sequences=[U, child_h, child_exists])
                return sub_h_f.sum(axis=0)

            h_f, _ = theano.map(
                fn=lambda U: _sub_f(U),
                sequences=[self.U_f])
            f = (T.nnet.sigmoid(
                    T.dot(self.W_f, parent_x).dimshuffle('x', 0) + h_f +
                    self.b_f.dimshuffle('x', 0)) *
                 child_exists.dimshuffle(0, 'x'))

            c = i * u + T.sum(f * child_c, axis=0)
            h = o * T.tanh(c)
            return h, c

        return unit
