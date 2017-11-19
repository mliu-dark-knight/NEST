from __future__ import print_function

import random
from collections import defaultdict, OrderedDict

import dill
from tqdm import tqdm

from GCN import *


class Graph(object):
	def __init__(self, params):
		self.params = params
		self.data_dir = 'data/' + self.params.dataset + '/'
		path = self.data_dir + 'graph.pkl'
		try:
			self.nbs = dill.load(open(path, 'rb'))
		except:
			self.init_nbs()
			dill.dump(self.nbs, open(path, 'wb'))
		self.num_node = len(self.nbs)
		self.feature = self.read_feature()

	def init_nbs(self):
		self.nbs = defaultdict(lambda : set())
		with open(self.data_dir + self.params.graph, 'r') as f:
			next(f)
			for line in f:
				[n1, n2] = list(map(int, line.rstrip().split()))
				self.nbs[n1].add(n2)
				self.nbs[n2].add(n1)

	def subgraph_es(self, ns):
		ns_set = set(ns)
		es = []
		for n in ns:
			nbs = self.nbs[n] & ns_set
			es += [(n, nb) for nb in nbs]
		return es

	def read_feature(self):
		feature = []
		with open(self.data_dir + self.params.feature) as f:
			for line in f:
				feature.append(np.array(list(map(int, line.strip().split()))).astype(np.float32))
		#0 index indicates invalid node
		feature = [np.zeros(len(feature[0]))] + feature
		return np.array(feature)

class SubGraph(object):
	def __init__(self, ns, path):
		self.ns = ns
		self.init(path)

	def init(self, path):
		self.kernels = []
		kernel = np.expand_dims(np.array(self.ns), axis=1)
		num = 1
		with open(path, 'r') as f:
			for line in f:
				line = line.rstrip().split()
				if line[0] == '#':
					if len(kernel) == 0:
						kernel = np.expand_dims(np.array([0] * num), axis=0)
					self.kernels.append(np.array(kernel))
					num = int(line[2])
					kernel = []
				else:
					# python set is ordered
					unique = list(OrderedDict.fromkeys(line))
					kernel.append(np.array(list(map(int, unique))))
		if len(kernel) == 0:
			kernel = np.expand_dims(np.array([0] * num), axis=0)
		self.kernels.append(np.array(kernel))


class Data(object):
	def __init__(self, subgraph, label):
		self.subgraph = subgraph
		self.label = label

'''
train.pkl and test.pkl are pre-serialized training data and test data,
remember to delete them after running preprocess.py
'''
class Predictor(object):
	def __init__(self, params):
		self.params = params
		self.data_dir = 'data/' + self.params.dataset + '/'
		self.graph = Graph(params)
		data_dump = self.data_dir + 'data.pkl'
		try:
			self.data = dill.load(open(data_dump, 'rb'))
		except:
			self.data = self.read_data()
			dill.dump(self.data, open(data_dump, 'wb'))
		self.kernel_sizes = [len(kernel[0]) for kernel in self.data[0].subgraph.kernels]
		self.num_kernel = len(self.kernel_sizes)
		self.num_label = len(self.data[0].label)

	def read_data(self):
		data = []
		id = 0
		with open(self.data_dir + self.params.data, 'r') as f1:
			with open(self.data_dir + self.params.label, 'r') as f2:
				for line1, line2 in zip(f1, f2):
					cascade = list(map(int, line1.strip().split()))
					ns = cascade
					subgraph = SubGraph(ns, self.data_dir + self.params.meta + 'g' + str(id))
					label = np.array(list(map(int, line2.strip().split())))
					id += 1
					data.append(Data(subgraph, label))
		return data

	def feed_dict(self, data, training):
		subgraph, label = data.subgraph, data.label
		feed = {k: kernel for k, kernel in zip(self.model.kernel, subgraph.kernels)}
		feed[self.model.label] = label
		feed[self.model.training] = training
		return feed

	def fit(self):
		self.params.num_node = self.graph.num_node
		self.params.kernel_sizes = self.kernel_sizes
		self.params.num_kernel = self.num_kernel
		self.params.num_label = self.num_label
		random.shuffle(self.data)
		split_idx = int(len(self.data) / 10)
		train, test = self.data[:-split_idx], self.data[-split_idx:]
		print('Start training')
		with tf.Session() as sess:
			self.model = GCN(self.params, self.graph)
			sess.run(tf.global_variables_initializer())
			for _ in tqdm(range(self.params.epoch), ncols=100):
				for i in tqdm(range(len(train)), ncols=100):
					data = train[i]
					sess.run(self.model.gradient_descent, feed_dict=self.feed_dict(data, True))
			train_accuracy = self.eval(sess, train)
			test_accuracy = self.eval(sess, test)
			print('Training Accuracy: %f', train_accuracy)
			print('Testing Accuracy: %f', test_accuracy)

	def eval(self, sess, test):
		correct = 0.0
		for i in tqdm(range(len(test)), ncols=100):
			data = test[i]
			truth = np.where(data.label == 1)[0][0]
			predict = sess.run(self.model.predict, feed_dict=self.feed_dict(data, False))
			if predict == truth:
				correct += 1
		return correct / len(test)
