__doc__ = """Tree RNNs aka Recursive Neural Networks."""

import numpy as np
import theano
from theano import tensor as T


theano.config.floatX = 'float32'


class Node(object):
    def __init__(self, val=None):
        self.children = []
        self.val = val
        self.idx = None

    def add_children(self, other_children):
        self.children.extend(other_children)


def gen_nn_inputs(root_node):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    x = _get_leaf_vals(root_node)
    tree = _get_tree_traversal(root_node, len(x))
    return (np.array(x, dtype='int32'),
            np.array(tree, dtype='int32'))


def _get_leaf_vals(root_node):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if not node.children:
                all_leaves.append(node)
            else:
                next_layer.extend(node.children[::-1])
        layer = next_layer

    vals = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        vals.append(leaf.val)
    return vals


def _get_tree_traversal(root_node, start_idx=0):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return []
    tree = []
    for child in root_node.children:
        subtree = _get_tree_traversal(child, start_idx)
        tree.extend(subtree)
        start_idx += len(subtree)
    child_idxs = [child.idx for child in root_node.children]
    assert not any(idx is None for idx in child_idxs)
    root_node.idx = start_idx
    tree.append(child_idxs + [root_node.idx])
    return tree


class TreeRNN(object):

    def __init__(self, num_emb, emb_dim, output_dim, degree=2, learning_rate=0.01):
        assert emb_dim > 1
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.degree = degree
        self.learning_rate = learning_rate

        self.params = []
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        self.recursive_unit = self.create_recursive_unit()
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.emb_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.embeddings, self.W_out, self.b_out])

        self.x = T.ivector(name='x')  # word indices
        self.tree = T.imatrix(name='tree')  # shape [None, self.degree + 1]
        self.y = T.fvector(name='y')  # output shape [self.output_dim]

        num_words = self.x.shape[0]  # also num leaves
        num_nodes = self.tree.shape[0]  # num internal nodes
        emb_x = self.embeddings[self.x]

        def _recurrence(node_info, t, node_emb, last_emb):
            child_emb = node_emb[node_info - t]
            parent_emb = self.recursive_unit(child_emb)
            node_emb = T.concatenate([node_emb,
                                      parent_emb.reshape([1, self.emb_dim])])
            return node_emb[1:], parent_emb

        dummy = theano.shared(self.init_vector([self.emb_dim]))
        (_, parent_embeddings), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[emb_x, dummy],
            sequences=[self.tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        self.final_state = parent_embeddings[-1]

        self.pred_y = self.activation(
            T.dot(self.W_out, self.final_state) + self.b_out)
        self.loss = self.loss_fn(self.y, self.pred_y)

        self.grad = T.grad(self.loss, self.params)
        updates = [(param, param - self.learning_rate * grad)
                   for param, grad in zip(self.params, self.grad)]

        self.train = theano.function([self.x, self.tree, self.y],
                                     [self.loss, self.pred_y],
                                     updates=updates)

        self.evaluate = theano.function([self.x, self.tree],
                                        self.final_state)

    def train_step_inner(self, x, tree, y):
        assert np.array_equal(tree[:, -1], np.arange(len(x), len(x) + len(tree)))
        assert np.all(tree[:, 0] + 1 >= np.arange(len(tree)))
        assert np.all(tree[:, 1] + 1 >= np.arange(len(tree)))
        return self.train(x, tree[:, :-1], y)

    def train_step(self, root_node, y):
        x, tree = gen_nn_inputs(root_node)
        return self.train_step_inner(x, tree, y)

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)

    def create_recursive_unit(self):
        self.W_rec = theano.shared(self.init_matrix([self.emb_dim, self.emb_dim]))
        self.h_rec = theano.shared(self.init_vector([self.emb_dim]))
        self.params.extend([self.W_rec, self.h_rec])
        def unit(child_emb):  # very simple
            mean_emb = T.mean(child_emb, axis=0)
            next_emb = T.tanh(self.h_rec + T.dot(self.W_rec, mean_emb))
            return next_emb
        return unit

    def activation(self, inp):
        return T.nnet.sigmoid(inp)

    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))
