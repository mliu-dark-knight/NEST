from Predictor import Predictor
import flags

flags.DEFINE_string('dataset', 'cora', '[cora, citeseer]')
flags.DEFINE_string('subgraph', 'subgraph/', 'Directory of all subgraphs, each file is a subgraph')
flags.DEFINE_string('graph', 'graph.txt', 'Edge list of the complete graph')
flags.DEFINE_string('kernel', 'kernel.json', 'Kernels to be matched')
flags.DEFINE_string('query', 'query', 'Used to create query files used by SubMatch')
flags.DEFINE_string('meta', 'meta/', 'Directory of matched instances of kernels')
flags.DEFINE_string('data', 'data.txt', None)
flags.DEFINE_string('feature', 'feature.txt', None)
flags.DEFINE_string('label', 'label.txt', None)

flags.DEFINE_integer('feat_dim', -1, None)
flags.DEFINE_list('node_dim', [512, 256], 'Dimension of hidden layers between feature and node embedding')
flags.DEFINE_list('instance_h_dim', [256], 'Dimension of hidden layers between node embedding and instance embedding, last element is the dimension of instance embedding')
flags.DEFINE_list('graph_h_dim', [128], 'Dimension of hidden layers between instance embedding and subgraph embedding, last element is the dimension of subgraph embedding')
flags.DEFINE_float('keep_prob', 0.8, 'Used for dropout')

flags.DEFINE_list('kernel_sizes', [1], 'List of number of nodes in kernel')
flags.DEFINE_string('pooling', 'max', '[max, average, sum]')

flags.DEFINE_integer('epoch', 2, None)
flags.DEFINE_float('learning_rate', 1e-4, None)
flags.DEFINE_float('lambda_r', 0.1, 'Coefficient of reconstruction loss')
flags.DEFINE_float('memory_fraction', 1.0, None)

FLAGS = flags.FLAGS

if __name__ == '__main__':
	predictor = Predictor(FLAGS)
	train_accuracy,test_accuracy = predictor.fit()
	print('Training Accuracy: %f', train_accuracy)
	print('Testing Accuracy: %f', test_accuracy)
