from main import FLAGS
from Predictor import Graph
from threading import Thread

class Preprocess(object):
	def __init__(self, params=FLAGS):
		self.params = params
		self.graph = Graph(params)
		self.train_subg = self.read_subgraph(self.params.data_dir + 'train/' + self.params.train)
		self.test_subg = self.read_subgraph(self.params.data_dir + 'test/' + self.params.test)


	def read_subgraph(self, path):
		subgraph = []
		with open(path, 'r') as f:
			for line in f:
				src, rest = line.strip().split(1)
				subgraph.append([src] + rest.split()[::2][:-1])
		return subgraph

	def create_subgraph(self, subgraph, num_thread=10):
		def worker(id):
			counter = 0
			for i in xrange(subgraph):
				if counter % num_thread == id:
					self.write_subg(self.subgraph[i], i)

		threads = []
		for i in xrange(num_thread):
			thread = Thread(target=worker, args=(i, ))
			threads.append(thread)
			thread.start()
		for thread in threads:
			thread.join()

	def write_subg(self, ns, path, id):
		ns = sorted(ns)
		es = self.graph.subgraph_es(ns)
		with open(path + 'g' + str(id, 'w')) as f:
			f.write('t #\n')
			for n in ns:
				f.write('v %d 0\n' & n)
			for e in es:
				f.write('e %d %d 0\n' & (e[0], e[1]))

if __name__ == '__main__':
	preproc = Preprocess()
	preproc.create_subgraph(preproc.train_subg)
	preproc.create_subgraph(preproc.test_subg)
