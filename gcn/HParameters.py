import argparse
# Settings
desc = "Tensorflow implementation of (GCN)"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--dataset', default='cora', help='Dataset string.', type=str)  # 'cora', 'citeseer', 'pubmed'
parser.add_argument('--model', default='gcn', help='Model string.', type=str)  # 'gcn', 'gcn_cheby', 'dense'
parser.add_argument('--learning_rate', default=0.01, help='Initial learning rate.', type=float)
parser.add_argument('--epochs', default=200, help='Number of epochs to train.', type=int)
parser.add_argument('--hidden1', default=16, help='Number of units in hidden layer 1.', type=int)
parser.add_argument('--dropout', default=0.5, help='Dropout rate (1 - keep probability).', type=float)
parser.add_argument('--weight_decay', default=5e-4, help='Weight for L2 loss on embedding matrix.', type=float)
parser.add_argument('--early_stopping', default=10, help='Tolerance for early stopping (# of epochs).', type=int)
parser.add_argument('--max_degree', default=3, help='Maximum Chebyshev polynomial degree.', type=int)
FLAGS = parser.parse_args()