# tree_rnn
Theano implementation of Tree RNNs aka Recursive Neural Networks.

Includes implementation of TreeLSTMs as described in "Improved
Semantic Representations From Tree-Structured Long Short-Term
Memory Networks" by Kai Sheng Tai, Richard Socher, and Christopher
D. Manning.

Also includes implementation of TreeGRUs derived using similar
methods.

You may immediately run "dummy" demos via simple_demo.py and
modulo_demo.py.

Code for evaluation on the Stanford Sentiment Treebank (used by
the paper) is also available in sentiment.py.  To run this, you'll
need to download the relevant data.

Step-by-step for cloning this repo and getting the sentiment model
running:

From your shell, run

    git clone https://github.com/ofirnachum/tree_rnn.git
    git clone https://github.com/stanfordnlp/treelstm.git
    cd treelstm
    ./fetch_and_preprocess.sh

This will download the datasets, the word vectors, and do some
preprocessing on the data.  Once this is complete, go into the
tree_rnn directory and start a Python shell.  In that shell,
we'll preprocess the word vectors:

    import data_utils
    vocab = data_utils.Vocab()
    vocab.load('../treelstm/data/sst/vocab-cased.txt')
    words, embeddings = \
        data_utils.read_embeddings_into_numpy(
            '../treelstm/data/glove/glove.840B.300d.txt', vocab=vocab)

    import numpy as np
    np.save('../treelstm/data/words.npy', words)
    np.save('../treelstm/data/glove.npy', embeddings)

After exiting the Python shell, you can run the sentiment training
directly

    python sentiment.py

The first couple lines of output should be

    train 6920
    dev 872
    test 1821
    num emb 21701
    num labels 3
    epoch 0
    avg loss 16.7419t example 6919 of 6920
    dev score 0.586009174312
    epoch 1
    avg loss 13.8955t example 6919 of 6920
    dev score 0.69495412844
    epoch 2
    avg loss 12.9191t example 6919 of 6920
    dev score 0.730504587156
