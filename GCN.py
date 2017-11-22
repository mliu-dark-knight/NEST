import numpy as np
from NN import *

class GCN(object):
	def __init__(self, params, graph=None):
		self.params = params
		self.build(graph)

	def build_placeholder(self):
		self.training = tf.placeholder(tf.bool)
		'''
			placeholder can only take one subgraph to avoid multiple None dimensions
			tensorflow will throw exception when sub-dimensions does not match
			however, every subgraph can have different number of kernels
			therefore cannot pad null tensor otherwise would be bad for average pooling
			'''
		self.kernel = [tf.placeholder(tf.int32, [None, size]) for size in self.params.kernel_sizes]
		self.label = tf.placeholder(tf.int32, shape=[self.params.num_label])

	def build(self, graph):
		self.build_placeholder()

		K = tf.get_variable('global_K', [self.params.instance_h_dim[-1], 1],
		                    initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / self.params.instance_h_dim[-1])))

		if self.params.use_feature:
			feature = tf.Variable(graph.feature, trainable=False, dtype=tf.float32)
			hidden = feature
			for h, dim in enumerate(self.params.node_dim):
				hidden = dropout(fully_connected(hidden, dim, 'Encode_' + str(h)), self.params.keep_prob, self.training)
			self.embedding = hidden

			for h, dim in enumerate(reversed(self.params.node_dim[1:])):
				hidden = dropout(fully_connected(hidden, dim, 'Decode_' + str(h)), self.params.keep_prob, self.training)
			reconstruct = fully_connected(hidden, self.params.feat_dim, 'Decode_' + str(len(self.params.node_dim)))
			self.loss_r = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=feature, logits=reconstruct), axis=1))
		else:
			self.embedding = dropout(embedding('Embedding', [graph.num_node + 1, self.params.node_dim[-1]]), self.params.keep_prob, self.training)

		instance_embed = [tf.reshape(tf.nn.embedding_lookup(self.embedding, self.kernel[i]), [-1, self.params.kernel_sizes[i] * self.params.node_dim[-1]])
		                       for i in range(self.params.num_kernel)]
		for h, dim in enumerate(self.params.instance_h_dim):
			instance_embed = [fully_connected(instance_embed[i], dim, 'Conv_' + str(i) + str(h))
							  for i in range(self.params.num_kernel)]
			instance_embed = [dropout(embed, self.params.keep_prob, self.training) for embed in instance_embed]

		pooling = {'max': (lambda embed: tf.reduce_max(embed, axis=0)),
				   'average': (lambda embed: tf.reduce_mean(embed, axis=0)),
				   'sum': (lambda embed: tf.reduce_sum(embed, axis=0))}
		kernel_embed = tf.stack([pooling[self.params.pooling](embed) for embed in instance_embed])
		Q = fully_connected(kernel_embed, self.params.instance_h_dim[-1], 'Q', activation='linear')
		V = fully_connected(kernel_embed, self.params.instance_h_dim[-1], 'V', activation='linear')
		Q = tf.nn.softmax(Q / tf.sqrt(np.float32(self.params.instance_h_dim[-1])), dim=0)
		graph_embed = tf.matmul(tf.nn.softmax(tf.matmul(Q, K)), V, transpose_a=True)

		for h, dim in enumerate( self.params.graph_h_dim):
			graph_embed = dropout(fully_connected(graph_embed, dim, 'FC_' + str(h)), self.params.keep_prob, self.training)

		logits = fully_connected(graph_embed, self.params.num_label, 'logits', activation='linear')

		loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.expand_dims(self.label, axis=0))\
			   + self.params.lambda_2 * tf.add_n(tf.get_collection('l2'))

		if self.params.use_feature:
			loss += self.params.lambda_r * self.loss_r

		self.predict = tf.cast(tf.argmax(logits, 1), 'int32')

		global_step = tf.Variable(0, trainable=False)

		optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
		self.gradient_descent = optimizer.minimize(loss, global_step=global_step)

		# for variable in tf.trainable_variables():
		# 	print(variable.name, variable.get_shape())
