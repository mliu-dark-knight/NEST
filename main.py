from Predictor import *
import flags

flags.DEFINE_string('data_dir', 'cascade-datasets/twitter/', None)
flags.DEFINE_string('subgraph', 'subgraph/', None)
flags.DEFINE_string('graph', 'graph.txt', None)
flags.DEFINE_string('kernel', 'kernel.json', None)
flags.DEFINE_string('query', 'query', None)
flags.DEFINE_string('meta', 'meta/', None)
flags.DEFINE_string('train', 'train.txt', None)
flags.DEFINE_string('test', 'test.txt', None)

flags.DEFINE_integer('num_node', 1, None)
flags.DEFINE_integer('node_dim', 100, 'for information cascade, dimension of node embedding must equal to dimension of subgraph embedding')
flags.DEFINE_list('instance_h_dim', [200, 100], 'dimension of hidden layers between node embedding and instance embedding, last element is the dimension of instance embedding')
flags.DEFINE_list('instance_activation', ['lrelu'] * 2, 'activation function for each hidden layer [sigmoid, tanh, relu, lrelu, elu]')
flags.DEFINE_list('graph_h_dim', [200, 100], 'dimension of hidden layers between instance embedding and subgraph embedding, last element is the dimension of subgraph embedding')
flags.DEFINE_list('graph_activation', ['elu'] * 2, 'activation function for each hidden layer [sigmoid, tanh, relu, lrelu, elu]')

flags.DEFINE_integer('num_kernel', 1, None)
flags.DEFINE_list('kernel_sizes', [1], 'list of number of nodes in kernel, length of list must be equal to num_kernel')
flags.DEFINE_string('pooling', 'sum', 'max pooling or average pooling [max, average, sum]')

flags.DEFINE_integer('epoch', 1, None)
flags.DEFINE_float('learning_rate', 0.01, None)
flags.DEFINE_float('decay_rate', 0.0, None)
flags.DEFINE_integer('k', 10, 'k in top_k')
flags.DEFINE_string('model', 'GCN', None)

FLAGS = flags.FLAGS

if __name__ == '__main__':
	# with tf.Session() as sess:
	# 	model = eval(FLAGS.model)(FLAGS)
	# 	sess.run(tf.global_variables_initializer())

	predictor = Predictor(FLAGS)
	predictor.fit()
