from __future__ import print_function

import pickle
from collections import OrderedDict

from tqdm import tqdm

from GCN import *


class Graph(object):
	def __init__(self, ns=None, path=None, kernels=None):
		# ns are indexed from 1
		self.ns = ns
		if kernels:
			self.kernels = kernels
		else:
			self.init(path)

	def init(self, path):
		self.kernels = []
		kernel = np.expand_dims(np.array([n + 1 for n in self.ns]), axis=1)
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
					kernel.append(np.array(list(map(lambda n: int(n) + 1, unique))))
		if len(kernel) == 0:
			kernel = np.expand_dims(np.array([0] * num), axis=0)
		self.kernels.append(np.array(kernel))


class Data(object):
	def __init__(self, graph, label):
		self.graph = graph
		self.label = label


class Classifier(object):
	def __init__(self, params):
		self.params = params
		self.data_dir = 'data/' + self.params.dataset + '/'
		self.data = self.read_data()
		self.kernel_sizes = [len(kernel[0]) for kernel in self.data[0].graph.kernels]
		self.num_kernel = len(self.kernel_sizes)

	def read_data(self):
		f = open(self.data_dir + self.params.dataset + '.graph')
		graphs = pickle.loads(f.read())
		f.close()

		self.num_label = 0
		self.label_map = {}
		labels = np.squeeze(graphs['labels'])
		for label in labels:
			if label not in self.label_map:
				self.label_map[label] = self.num_label
				self.num_label += 1

		data = []
		self.num_feat = 1
		self.feat_map = {}
		for id, raw_graph in graphs['graph'].items():
			graph = Graph(ns=range(len(raw_graph)), path=self.data_dir + self.params.meta + 'g' + str(id))
			graph = self.normalize_graph(graph, raw_graph)
			data.append(Data(graph, self.label_map[labels[id]]))
		return data

	def map_data_to_feat(self, graph):
		data_to_feat = {0: 0}
		for n, info in graph.items():
			# label 0 reserved for null
			label = info['label']
			if label not in self.feat_map:
				self.feat_map[label] = self.num_feat
				self.num_feat += 1
			data_to_feat[n + 1] = self.feat_map[label]
		return data_to_feat

	def normalize_graph(self, graph, raw_graph):
		data_to_feat = self.map_data_to_feat(raw_graph)
		kernels = []
		for kernel in graph.kernels:
			norm_kernel = []
			for instance in kernel:
				norm_kernel.append(np.array([data_to_feat[n] for n in instance]))
			kernels.append(np.array(norm_kernel))
		return Graph(kernels=kernels)

	def feed_dict(self, data):
		graph, label = data.graph, data.label
		feed = {k: kernel for k, kernel in zip(self.model.kernel, graph.kernels)}
		feed[self.model.label] = label
		return feed

	def fit(self):
		self.params.num_node = self.num_feat
		self.params.kernel_sizes = self.kernel_sizes
		self.params.num_kernel = self.num_kernel
		self.params.num_label = self.num_label
		print('Start training')
		with tf.Session() as sess:
			self.model = GCN(self.params)
			sess.run(tf.global_variables_initializer())
			for _ in tqdm(range(self.params.epoch), ncols=100):
				for i in tqdm(range(len(self.data)), ncols=100):
					data = self.data[i]
					sess.run(self.model.gradient_descent, feed_dict=self.feed_dict(data))
			accuracy = self.eval(sess)
			print('Accuracy: %f', accuracy)

	def eval(self, sess):
		correct = 0.0
		all_data = self.data
		for data in all_data:
			feed_dict = self.feed_dict(data)
			truth = data.label
			predict = sess.run(self.model.predict, feed_dict=feed_dict)
			if predict == truth:
				correct += 1.0
		return correct / len(all_data)
