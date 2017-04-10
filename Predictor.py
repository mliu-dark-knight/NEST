import dill
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from GCN import *

class Graph(object):
	def __init__(self, params):
		self.params = params
		if os.path.exists(self.params.data_dir + 'graph.pkl'):
			self.nbs = dill.load(open(self.params.data_dir + 'graph.pkl', 'rb'))
		else:
			self.init_nbs()
			dill.dump(self.nbs, open(self.params.data_dir + 'graph.pkl', 'wb'))

	def init_nbs(self):
		self.nbs = defaultdict(lambda : set())
		with open(self.params.data_dir + self.params.graph, 'r') as f:
			next(f)
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
		self.kernels.append(kernel)


class Data(object):
	def __init__(self, subgraph, candidate, next):
		self.subgraph = subgraph
		self.candidate = candidate
		self.next = next


class Predictor(object):
	def __init__(self, params):
		self.params = params
		# self.graph = Graph(params)
		self.train = self.read_data('train')
		self.test = self.read_data('test')

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
				id += 1
				data.append(Data(subgraph, candidate, next))
		return data

	def feed_dict(self, data):
		subgraph, candidate, next = data.subgraph, data.candidate, data.next
		feed = {k: kernel for k, kernel in zip(self.model.kernel, subgraph.kernels)}
		feed[self.model.candidate] = candidate
		feed[self.model.next] = next
		return feed

	def train(self):
		with tf.Session() as sess:
			self.model = eval(self.params.model)(self.params)
			sess.run(tf.global_variables_initializer())
			for _ in tqdm(range(self.params.epoch), ncols=100):
				for data in self.train:
					sess.run(self.model.gradient_descent, feed_dict=self.feed_dict(data))


	def eval(self, mode):
		pass

