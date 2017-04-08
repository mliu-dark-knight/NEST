from collections import defaultdict
from GCN import GCN

class Graph(object):
	def __init__(self, params):
		self.params = params
		self.init_nbs()

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
