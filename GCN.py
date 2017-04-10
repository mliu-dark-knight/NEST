from NN import *

class GCN(object):
	def __init__(self, params):
		self.params = params
		self.build()

	def build(self):
		self.embedding = embedding('embedding', [self.params.num_node, self.params.dim])
		'''
		placeholder can only take one subgraph to avoid multiple None dimensions
		tensorflow will throw exception when sub-dimensions does not match
		however, every subgraph can have different number of kernels
		therefore cannot pad null tensor otherwise would be bad for average pooling
		'''
		self.kernel = [tf.placeholder(tf.int32, [None, size]) for size in self.params.kernel_sizes]
		self.candidate = tf.placeholder(tf.int32, [None])
		# indicate which entry is the ground truth
		self.next = tf.placeholder(tf.int32, [1])


		kernel_embed = [tf.reshape(tf.nn.embedding_lookup(self.embedding, self.kernel[i]), [-1, self.params.kernel_sizes[i] * self.params.dim])
		                       for i in xrange(self.params.num_kernel)]

		kernel_conv = [fully_connected(kernel_embed[i], self.params.dim, 'Conv' + str(i), activation='relu')
		                    for i in xrange(self.params.num_kernel)]

		kernel_pool = [tf.reduce_max(conv, axis=0) if self.params.pooling == 'max' else tf.reduce_mean(conv, axis=0)
		               for conv in kernel_conv]
		concat = tf.expand_dims(tf.concat(kernel_pool, axis=0), dim=0)

		graph_embed = fully_connected(concat, self.params.fc_dim, 'FC1', activation='tanh')
		graph_embed = fully_connected(graph_embed, self.params.dim, 'FC2', activation=None)

		candidate_embed = tf.nn.embedding_lookup(self.embedding, self.candidate)

		logits = tf.matmul(candidate_embed, tf.transpose(graph_embed))
		self.softmax = tf.nn.softmax(logits)
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.expand_dims(self.next, dim=0), logits=tf.expand_dims(logits, dim=0))

		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.inverse_time_decay(self.params.learning_rate, global_step, 1, self.params.decay_rate)

		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.gradient_descent = optimizer.minimize(loss, global_step=global_step)

		for variable in tf.trainable_variables():
			print(variable.name, variable.get_shape())
