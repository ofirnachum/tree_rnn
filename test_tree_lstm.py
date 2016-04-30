import tree_lstm


def test():
    # very simple for now... just checks compilation
    # TODO: better tests
    model = tree_lstm.ChildSumTreeLSTM(10, 20, 30, 1)
