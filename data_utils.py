__doc__ = """Utilities for loading language datasets.

Basically porting http://github.com/stanfordnlp/treelstm/tree/master/util to Python.

"""

import tree_rnn

import numpy as np
import os


def read_sentiment_dataset(data_dir, fine_grained=False, dependency=False):
    vocab = Vocab()
    vocab.load(os.path.join(data_dir, 'vocab-cased.txt'))

    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    test_dir = os.path.join(data_dir, 'test')

    data = {}
    overall_max_degree = 0
    for name, sub_dir in zip(['train', 'dev', 'test'], [train_dir, dev_dir, test_dir]):
        if dependency:
            max_degree, trees = read_trees(
                os.path.join(sub_dir, 'dparents.txt'),
                os.path.join(sub_dir, 'dlabels.txt'))
        else:
            max_degree, trees = read_trees(
                os.path.join(sub_dir, 'parents.txt'),
                os.path.join(sub_dir, 'labels.txt'))
        sentences = read_sentences(
            os.path.join(sub_dir, 'sents.txt'),
            vocab)

        this_dataset = zip(trees, sentences)
        if not fine_grained:  # remove all 'neutral' data
            this_dataset = [(tree, sentence) for tree, sentence in this_dataset
                            if tree.label != 0]

        for tree, sentence in this_dataset:
            _remap_tokens_and_labels(tree, sentence, fine_grained)

        data[name] = [(tree, tree.label) for tree, _ in this_dataset]
        overall_max_degree = max(overall_max_degree, max_degree)

    data['max_degree'] = overall_max_degree
    assert overall_max_degree == 2 or dependency
    return vocab, data


class Vocab(object):

    def __init__(self):
        self.words = []
        self.word2idx = {}
        self.unk_index = None
        self.start_index = None
        self.end_index = None
        self.unk_token = None
        self.start_token = None
        self.end_token = None

    def load(self, path):
        with open(path, 'r') as in_file:
            for line in in_file:
                word = line.strip()
                assert word not in self.word2idx
                self.word2idx[word] = len(self.words)
                self.words.append(word)

        for unk in ['<unk>', '<UNK>', 'UUUNKKK']:
            self.unk_index = self.unk_index or self.word2idx.get(unk, None)
            if self.unk_index is not None:
                self.unk_token = unk
                break

        for start in ['<s>', '<S>']:
            self.start_index = self.start_index or self.word2idx.get(start, None)
            if self.start_index is not None:
                self.start_token = start
                break

        for end in ['</s>', '</S>']:
            self.end_index = self.end_index or self.word2idx.get(end, None)
            if self.end_index is not None:
                self.end_token = end
                break

    def index(self, word):
        if self.unk_index is None:
            assert word in self.word2idx
        return self.word2idx.get(word, self.unk_index)

    def size(self):
        return len(self.words)


def read_trees(parents_file, labels_file):
    trees = []
    max_degree = 0
    with open(parents_file, 'r') as parents_f:
        with open(labels_file, 'r') as labels_f:
            while True:
                cur_parents = parents_f.readline()
                cur_labels = labels_f.readline()
                if not cur_parents or not cur_labels:
                    break
                cur_parents = [int(p) for p in cur_parents.strip().split()]
                cur_labels = [int(l) if l != '#' else None for l in cur_labels.strip().split()]
                cur_max_degree, cur_tree = read_tree(cur_parents, cur_labels)
                max_degree = max(max_degree, cur_max_degree)
                trees.append(cur_tree)
    return max_degree, trees


def read_tree(parents, labels):
    nodes = {}
    parents = [p - 1 for p in parents]  # 1-indexed
    for i in xrange(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tree_rnn.Node(val=idx)  # for now, val is just idx
                if prev is not None:
                    assert prev.val != node.val
                    node.add_child(prev)

                node.label = labels[idx]
                nodes[idx] = node

                parent = parents[idx]
                if parent in nodes:
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break

                prev = node
                idx = parent

    # ensure tree is connected
    num_roots = sum(node.parent is None for node in nodes.itervalues())
    assert num_roots == 1, num_roots

    # overwrite vals to match sentence indices -
    # only leaves correspond to sentence tokens
    leaf_idx = 0
    for node in nodes.itervalues():
        if node.children:
            node.val = None
        else:
            node.val = leaf_idx
            leaf_idx += 1

    max_degree = max(len(node.children) for node in nodes.itervalues())

    return max_degree, root


def read_sentences(path, vocab):
    sentences = []
    with open(path, 'r') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            sentences.append([vocab.index(tok) for tok in tokens])
    return sentences


def _remap_tokens_and_labels(tree, sentence, fine_grained):
    # map leaf idx to word idx
    if tree.val is not None:
        tree.val = sentence[tree.val]

    # map label to suitable range
    if tree.label is not None:
        if fine_grained:
            tree.label += 2
        else:
            if tree.label < 0:
                tree.label = 0
            elif tree.label == 0:
                tree.label = 1
            else:
                tree.label = 2

    [_remap_tokens_and_labels(child, sentence, fine_grained)
     for child in tree.children
     if child is not None]


def read_embeddings_into_numpy(file_name, vocab=None):
    """Reads Glove vector files and returns numpy arrays.

    If vocab is given, only intersection of vocab and words is used.

    """
    words = []
    array = []
    with open(file_name, 'r') as in_file:
        for line in in_file:
            fields = line.strip().split()
            word = fields[0]
            if vocab and word not in vocab.word2idx:
                continue
            embedding = np.array([float(f) for f in fields[1:]])
            words.append(word)
            array.append(embedding)

    return np.array(words), np.array(array)
