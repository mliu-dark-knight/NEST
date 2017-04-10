from __future__ import print_function
import dill
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from GCN import *

class Graph(object):
	def __init__(self, params):
		self.params = params
		path = self.params.data_dir + 'graph.pkl'
		if os.path.exists(path):
			self.nbs = dill.load(open(path, 'rb'))
		else:
			self.init_nbs()
			dill.dump(self.nbs, open(path, 'wb'))
		self.num_node = len(self.nbs)

	def init_nbs(self):
		self.nbs = defaultdict(lambda : set())
		with open(self.params.data_dir + self.params.graph, 'r') as f:
			for line in f:
				[n1, n2] = map(int, line.rstrip().split())
				self.nbs[n1].add(n2)
				self.nbs[n2].add(n1)


	def subgraph_nbs(self, ns):
		nbs = [self.nbs[n] for n in ns]
		return set.union(*nbs)

	def subgraph_es(self, ns):
		ns_set = set(ns)
		es = []
		for n in ns:
			nbs = self.nbs[n] & ns_set
			es += [(n, nb) for nb in nbs]
		return es


class SubGraph(object):
	def __init__(self, ns, path):
		self.ns = ns
		self.init(path)

	def init(self, path):
		self.kernels = []
		kernel = np.array(self.ns)
		num = 1
		with open(path, 'r') as f:
			for line in f:
				line = line.rstrip().split()
				if line[0] == '#':
					if len(kernel) == 0:
						kernel = np.array([0] * num)
					self.kernels.append(kernel)
					num = int(line[1])
					kernel = []
				else:
					kernel.append(np.array(map(int, line)))
		self.kernels.append(np.array(kernel))


class Data(object):
	def __init__(self, subgraph, candidate, next):
		self.subgraph = subgraph
		self.candidate = candidate
		self.next = next


class Predictor(object):
	def __init__(self, params):
		self.params = params
		self.graph = Graph(params)
		train_path = self.params.data_dir + 'train/train.pkl'
		if os.path.exists(train_path):
			self.train = dill.load(open(train_path, 'rb'))
		else:
			self.train = self.read_data('train')
			dill.dump(self.train, open(train_path, 'wb'))
		test_path = self.params.data_dir + 'test/test.pkl'
		if os.path.exists(test_path):
			self.test = dill.load(open(test_path, 'rb'))
		else:
			self.test = self.read_data('test')
			dill.dump(self.test, open(test_path, 'wb'))
		self.kernel_sizes = [1] + [len(kernel[0]) for kernel in self.train[0].subgraph.kernels[1:]]
		self.num_kernel = len(self.kernel_sizes)

	def read_data(self, mode):
		data = []
		id = 0
		with open(self.params.data_dir + mode + '/' + getattr(self.params, mode), 'r') as f:
			for line in f:
				src, rest = line.strip().split(' ', 1)
				cascade = map(int, [src] + rest.split()[::2])
				ns, next = cascade[:-1], cascade[-1]
				subgraph = SubGraph(ns, self.params.data_dir + mode + '/' + self.params.meta + 'g' + str(id))
				candidate = self.graph.subgraph_nbs(ns) - set(ns)
				candidate.add(next)
				id += 1
				data.append(Data(subgraph, np.array(list(candidate)), next))
		return data

	def feed_dict(self, data):
		subgraph, candidate, next = data.subgraph, data.candidate, data.next
		feed = {k: kernel for k, kernel in zip(self.model.kernel, subgraph.kernels)}
		feed[self.model.candidate] = candidate
		feed[self.model.next] = np.where(candidate == next)[0][0]
		return feed

	def fit(self):
		self.params.num_node = self.graph.num_node
		self.params.kernel_sizes = self.kernel_sizes
		self.params.num_kernel = self.num_kernel
		print('Start training')
		with tf.Session() as sess:
			self.model = eval(self.params.model)(self.params)
			sess.run(tf.global_variables_initializer())
			for _ in tqdm(range(self.params.epoch), ncols=100):
				for data in self.train:
					sess.run(self.model.gradient_descent, feed_dict=self.feed_dict(data))
			print('Training accuracy: %f', self.eval('train', sess))
			print('Testing accuracy: %f', self.eval('test', sess))


	def eval(self, mode, sess):
		correct = 0.0
		all_data = self.train if mode == 'train' else self.test
		for data in all_data:
			predict = sess.run(self.model.predict, feed_dict=self.feed_dict(data))
			if predict == np.where(data.candidate == data.next)[0][0]:
				correct += 1.0
		return correct / len(all_data)
