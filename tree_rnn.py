__doc__ = """Tree RNNs aka Recursive Neural Networks."""

import numpy as np
import theano
from theano import tensor as T
from theano.compat.python2x import OrderedDict


theano.config.floatX = 'float32'


class Node(object):
    def __init__(self, val=None):
        self.children = []
        self.val = val
        self.idx = None
        self.height = 1
        self.size = 1
        self.num_leaves = 1
        self.parent = None
        self.label = None

    def _update(self):
        self.height = 1 + max([child.height for child in self.children if child] or [0])
        self.size = 1 + sum(child.size for child in self.children if child)
        self.num_leaves = (all(child is None for child in self.children) +
                           sum(child.num_leaves for child in self.children if child))
        if self.parent is not None:
            self.parent._update()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        self._update()

    def add_children(self, other_children):
        self.children.extend(other_children)
        for child in other_children:
            child.parent = self
        self._update()


class BinaryNode(Node):
    def __init__(self, val=None):
        super(BinaryNode, self).__init__(val=val)

    def add_left(self, node):
        if not self.children:
            self.children = [None, None]
        self.children[0] = node
        node.parent = self
        self._update()

    def add_right(self, node):
        if not self.children:
            self.children = [None, None]
        self.children[1] = node
        node.parent = self
        self._update()

    def get_left(self):
        if not self.children:
            return None
        return self.children[0]

    def get_right(self):
        if not self.children:
            return None
        return self.children[1]


def gen_nn_inputs(root_node, max_degree=None, only_leaves_have_vals=True,
                  with_labels=False):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    _clear_indices(root_node)
    x, leaf_labels = _get_leaf_vals(root_node)
    tree, internal_x, internal_labels = \
        _get_tree_traversal(root_node, len(x), max_degree)
    assert all(v is not None for v in x)
    if not only_leaves_have_vals:
        assert all(v is not None for v in internal_x)
        x.extend(internal_x)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)
    if with_labels:
        labels = leaf_labels + internal_labels
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(x, dtype='int32'),
                np.array(tree, dtype='int32'),
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX))
    return (np.array(x, dtype='int32'),
            np.array(tree, dtype='int32'))


def _clear_indices(root_node):
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


def _get_leaf_vals(root_node):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if all(child is None for child in node.children):
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    vals = []
    labels = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        vals.append(leaf.val)
        labels.append(leaf.label)
    return vals, labels


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer

    tree = []
    internal_vals = []
    labels = []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_vals.append(node.val if node.val is not None else -1)
            labels.append(node.label)
            idx += 1

    return tree, internal_vals, labels


class TreeRNN(object):
    """Data is represented in a tree structure.

    Every leaf and internal node has a data (provided by the input)
    and a memory or hidden state.  The hidden state is computed based
    on its own data and the hidden states of its children.  The
    hidden state of leaves is given by a custom init function.

    The entire tree's embedding is represented by the final
    state computed at the root.

    """

    def __init__(self, num_emb, emb_dim, hidden_dim, output_dim,
                 degree=2, learning_rate=0.01, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=False):
        assert emb_dim > 1 and hidden_dim > 1
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.degree = degree
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.params = []
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        if trainable_embeddings:
            self.params.append(self.embeddings)

        self.x = T.ivector(name='x')  # word indices
        self.tree = T.imatrix(name='tree')  # shape [None, self.degree]
        if labels_on_nonroot_nodes:
            self.y = T.fmatrix(name='y')  # output shape [None, self.output_dim]
            self.y_exists = T.fvector(name='y_exists')  # shape [None]
        else:
            self.y = T.fvector(name='y')  # output shape [self.output_dim]

        self.num_words = self.x.shape[0]  # total number of nodes (leaves + internal) in tree
        emb_x = self.embeddings[self.x]
        emb_x = emb_x * T.neq(self.x, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings

        self.tree_states = self.compute_tree(emb_x, self.tree)
        self.final_state = self.tree_states[-1]
        if labels_on_nonroot_nodes:
            self.output_fn = self.create_output_fn_multi()
            self.pred_y = self.output_fn(self.tree_states)
            self.loss = self.loss_fn_multi(self.y, self.pred_y, self.y_exists)
        else:
            self.output_fn = self.create_output_fn()
            self.pred_y = self.output_fn(self.final_state)
            self.loss = self.loss_fn(self.y, self.pred_y)

        updates = self.gradient_descent(self.loss)

        train_inputs = [self.x, self.tree, self.y]
        if labels_on_nonroot_nodes:
            train_inputs.append(self.y_exists)
        self._train = theano.function(train_inputs,
                                      [self.loss, self.pred_y],
                                      updates=updates)

        self._evaluate = theano.function([self.x, self.tree],
                                         self.final_state)

        self._predict = theano.function([self.x, self.tree],
                                        self.pred_y)

    def _check_input(self, x, tree):
        assert np.array_equal(tree[:, -1], np.arange(len(x) - len(tree), len(x)))
        if not self.irregular_tree:
            assert np.all((tree[:, 0] + 1 >= np.arange(len(tree))) |
                          (tree[:, 0] == -1))
            assert np.all((tree[:, 1] + 1 >= np.arange(len(tree))) |
                          (tree[:, 1] == -1))

    def train_step_inner(self, x, tree, y):
        self._check_input(x, tree)
        return self._train(x, tree[:, :-1], y)

    def train_step(self, root_node, y):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        return self.train_step_inner(x, tree, y)

    def evaluate(self, root_node):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        self._check_input(x, tree)
        return self._evaluate(x, tree[:, :-1])

    def predict(self, root_node):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        self._check_input(x, tree)
        return self._predict(x, tree[:, :-1])

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)

    def create_output_fn(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out])

        def fn(final_state):
            return T.nnet.softmax(
                T.dot(self.W_out, final_state) + self.b_out)
        return fn

    def create_output_fn_multi(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out])

        def fn(tree_states):
            return T.nnet.softmax(
                T.dot(tree_states, self.W_out.T) +
                self.b_out.dimshuffle('x', 0))
        return fn

    def create_recursive_unit(self):
        self.W_hx = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.W_hh = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([self.W_hx, self.W_hh, self.b_h])
        def unit(parent_x, child_h, child_exists):  # very simple
            h_tilde = T.sum(child_h, axis=0)
            h = T.tanh(self.b_h + T.dot(self.W_hx, parent_x) + T.dot(self.W_hh, h_tilde))
            return h
        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_matrix([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            return self.recursive_unit(leaf_x, dummy, dummy.sum(axis=1))
        return unit

    def compute_tree(self, emb_x, tree):
        self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        leaf_h, _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        if self.irregular_tree:
            init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
        else:
            init_node_h = leaf_h

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, last_h):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h = self.recursive_unit(cur_emb, child_h, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            return node_h[1:], parent_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)

    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))

    def loss_fn_multi(self, y, pred_y, y_exists):
        return T.sum(T.sum(T.sqr(y - pred_y), axis=1) * y_exists, axis=0)

    def gradient_descent(self, loss):
        """Momentum GD with gradient clipping."""
        grad = T.grad(loss, self.params)
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params, grad)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (5.0 / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - self.learning_rate * grad
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates
