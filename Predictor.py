from __future__ import print_function
import dill
import os
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from GCN import *


class Graph(object):
	def __init__(self, params):
		self.params = params
		self.data_dir = self.params.task + '-datasets/' + self.params.dataset + '/'
		path = self.data_dir + 'graph.pkl'
		try:
			self.nbs = dill.load(open(path, 'rb'))
		except:
			self.init_nbs()
			dill.dump(self.nbs, open(path, 'wb'))
		self.num_node = len(self.nbs)

	def init_nbs(self):
		self.nbs = defaultdict(lambda : set())
		with open(self.data_dir + self.params.graph, 'r') as f:
			next(f)
			for line in f:
				[n1, n2] = list(map(int, line.rstrip().split()))
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
	def __init__(self, subgraph, candidate, next):
		self.subgraph = subgraph
		self.candidate = candidate
		self.next = next

'''
train.pkl and test.pkl are pre-serialized training data and test data,
remember to delete them after running preprocess.py
'''
class Predictor(object):
	def __init__(self, params):
		self.params = params
		self.data_dir = self.params.task + '-datasets/' + self.params.dataset + '/'
		self.graph = Graph(params)
		train_path = self.data_dir + 'train/train.pkl'
		try:
			self.train = dill.load(open(train_path, 'rb'))
		except:
			self.train = self.read_data('train')
			dill.dump(self.train, open(train_path, 'wb'))
		test_path = self.data_dir + 'test/test.pkl'
		try:
			self.test = dill.load(open(test_path, 'rb'))
		except:
			self.test = self.read_data('test')
			dill.dump(self.test, open(test_path, 'wb'))
		self.kernel_sizes = [len(kernel[0]) for kernel in self.train[0].subgraph.kernels]
		self.num_kernel = len(self.kernel_sizes)

	def read_data(self, mode):
		data = []
		id = 0
		with open(self.data_dir + getattr(self.params, mode), 'r') as f:
			for line in f:
				src, rest = line.strip().split(' ', 1)
				cascade = list(map(int, [src] + rest.split()[::2]))
				ns, next = cascade[:-1], cascade[-1]
				subgraph = SubGraph(ns, self.data_dir + mode + '/' + self.params.meta + 'g' + str(id))
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
			for epoch in tqdm(range(self.params.epoch), ncols=100):
				for i in tqdm(range(len(self.train)), ncols=100):
					data = self.train[i]
					sess.run(self.model.gradient_descent, feed_dict=self.feed_dict(data))
			_, hit, map = self.eval('train', sess)
			print('Training Hit: %f', hit)
			print('Training MAP: %f', map)
			_, hit, map = self.eval('test', sess)
			print('Testing Hit: %f', hit)
			print('Testing MAP: %f', map)


	def eval(self, mode, sess):
		correct, hit, map = 0.0, 0.0, 0.0
		all_data = self.train if mode == 'train' else self.test
		for data in all_data:
			feed_dict = self.feed_dict(data)
			truth = np.where(data.candidate == data.next)[0][0]
			predict = sess.run(self.model.predict, feed_dict=feed_dict)
			if predict == truth:
				correct += 1.0
			_, top_k = sess.run(self.model.top_k, feed_dict=feed_dict)
			top_k = top_k[0]
			if truth in top_k:
				hit += 1.0
				map += 1.0 / (np.where(top_k == truth)[0][0] + 1)
		return correct / len(all_data), hit / len(all_data), map / len(all_data)
