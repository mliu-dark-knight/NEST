from LSTM import *
from GCN import *
import flags

flags.DEFINE_integer('num_node', 1024, 'Node with index 0 is NULL')
flags.DEFINE_integer('dim', 64, None)
flags.DEFINE_integer('max_seq_len', 16, None)

flags.DEFINE_integer('num_kernel', 10, None)
flags.DEFINE_list('kernel_sizes', range(1, 11, 1), 'list of number of nodes in kernel')
flags.DEFINE_string('pooling', 'average', 'max pooling or average pooling [max, average]')
flags.DEFINE_integer('fc_dim', 1024, 'dimension for fully connected layers after pooling')

flags.DEFINE_float('learning_rate', 0.01, None)
flags.DEFINE_float('decay_rate', 0.01, None)


flags.DEFINE_string('model', 'GCN', None)
FLAGS = flags.FLAGS

if __name__ == '__main__':
	with tf.Session() as sess:
		model = eval(FLAGS.model)(FLAGS)
		sess.run(tf.global_variables_initializer())
