import math
import tensorflow as tf


def weight(name, shape, init='he'):
	assert init == 'he' and len(shape) == 2
	var = tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / shape[0])))
	tf.add_to_collection('l2', tf.nn.l2_loss(var))
	return var

def bias(name, dim, initial_value=1e-2):
	return tf.get_variable(name, dim, initializer=tf.constant_initializer(initial_value))

def embedding(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(minval=-0.5 / shape[1], maxval=0.5 / shape[1]))

def batch_norm(x, prefix):
	with tf.variable_scope('BN'):
		inputs_shape = x.get_shape()
		axis = list(range(len(inputs_shape) - 1))
		param_shape = inputs_shape[-1:]

		beta = tf.get_variable(prefix + '_beta', param_shape, initializer=tf.constant_initializer(0.))
		gamma = tf.get_variable(prefix + '_gamma', param_shape, initializer=tf.constant_initializer(1.))
		batch_mean, batch_var = tf.nn.moments(x, axis)
		normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
	return normed


def fully_connected(input, num_neurons, name, activation='relu', bn=False):
	func = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, None: lambda l: l}
	W = weight(name + '_W', [input.get_shape().as_list()[1], num_neurons], init='he')
	if bn:
		l = batch_norm(tf.matmul(input, W), name)
	else:
		l = tf.matmul(input, W) + bias(name + '_b', num_neurons)
	return func[activation](l)
