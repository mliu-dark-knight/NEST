from NN import *


class LSTM(object):
	def __init__(self, params):
		self.params = params
		self.build()

	def build(self):
		self.embed = embedding('embedding', [self.params.num_node, self.params.dim])
		self.sequence = tf.placeholder(tf.int32, [None, self.params.max_seq_len])
		self.label = tf.placeholder(tf.int32, [None])

		lstm = tf.contrib.rnn.LSTMCell(self.params.dim)
		hidden, _ = tf.nn.dynamic_rnn(lstm, tf.nn.embedding_lookup(self.embed, self.sequence), dtype=tf.float32)

		logits = fully_connected(tf.unstack(hidden, axis=1)[-1], self.params.dim, 'logits', activation=None, bn=False)

		with tf.name_scope('Loss'):
			loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits))
		with tf.name_scope('Accuracy'):
			self.predict = tf.cast(tf.argmax(logits, 1), 'int32')
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.label), tf.float32))

		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.inverse_time_decay(self.params.learning_rate, global_step, 1, self.params.decay_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.gradient_descent = optimizer.minimize(loss, global_step=global_step)

		for variable in tf.trainable_variables():
			print(variable.name, variable.get_shape())

