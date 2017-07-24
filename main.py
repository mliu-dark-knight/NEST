from Predictor import *
import flags

flags.DEFINE_string('task', 'general', '[cascade, general]')
flags.DEFINE_string('data_dir', 'cascade-datasets/twitter/', '[cascade-datasets/dblp, cascade-datasets/twitter]')
flags.DEFINE_string('subgraph', 'subgraph/', 'directory of all subgraphs, each file is a subgraph')
flags.DEFINE_string('graph', 'graph.txt', 'edge list of the complete graph')
flags.DEFINE_string('kernel', 'kernel.json', 'kernels to be matched')
flags.DEFINE_string('query', 'query', 'used to create query files used by SubMatch')
flags.DEFINE_string('meta', 'meta/', 'directory of matched instances of kernels')
flags.DEFINE_string('train', 'train.txt', 'each data point is a subgraph, if task is cascade, the last node of the subgraph will be treated as label')
flags.DEFINE_string('test', 'test.txt', None)

flags.DEFINE_integer('num_node', 1, 'dummy parameter for debugging')
flags.DEFINE_integer('node_dim', 100, 'for information cascade, dimension of node embedding must equal to dimension of subgraph embedding')
flags.DEFINE_list('instance_h_dim', [200, 100], 'dimension of hidden layers between node embedding and instance embedding, last element is the dimension of instance embedding')
flags.DEFINE_list('instance_activation', ['lrelu'] * 2, 'activation function for each hidden layer [sigmoid, tanh, relu, lrelu, elu]')
flags.DEFINE_list('graph_h_dim', [200, 100], 'dimension of hidden layers between instance embedding and subgraph embedding, last element is the dimension of subgraph embedding')
flags.DEFINE_list('graph_activation', ['elu'] * 2, 'activation function for each hidden layer [sigmoid, tanh, relu, lrelu, elu]')

flags.DEFINE_integer('num_kernel', 1, 'dummy parameter for debugging')
flags.DEFINE_list('kernel_sizes', [1], 'list of number of nodes in kernel, length of list must be equal to num_kernel')
flags.DEFINE_string('pooling', 'sum', 'max pooling or average pooling [max, average, sum]')

flags.DEFINE_integer('epoch', 1, 'training epoch')
flags.DEFINE_float('learning_rate', 0.01, None)
flags.DEFINE_float('decay_rate', 0.0, None)
flags.DEFINE_integer('k', 10, 'k in top_k, for inference in information cascade')
flags.DEFINE_string('model', 'GCN', None)

FLAGS = flags.FLAGS

if __name__ == '__main__':
	# with tf.Session() as sess:
	# 	model = eval(FLAGS.model)(FLAGS)
	# 	sess.run(tf.global_variables_initializer())

	predictor = Predictor(FLAGS)
	predictor.fit()
