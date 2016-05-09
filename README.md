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

Code for evaluation on the Standford Sentiment Treebank (used by
the paper) is also available in sentiment.py.  To run this, you'll
need to download the relevant data.  You can do so by cloning the
treelstm git repo (https://github.com/stanfordnlp/treelstm) and
running it's initial fetch_and_preprocess.sh script.  After this,
you'll also need to prepare the Glove vectors as .npy files
using the data_utils.read_embeddings_into_numpy method.
