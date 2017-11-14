import json
import os
import pickle
from subprocess import call

from main import FLAGS


class Prepare(object):
	def __init__(self, params=FLAGS):
		self.params = params
		self.data_dir = 'data/' + self.params.dataset + '/'
		self.subgraphs = self.read_pickle(self.data_dir + self.params.dataset + '.graph')

	def read_pickle(self, path):
		f = open(path)
		data = pickle.loads(f.read())
		f.close()
		return data['graph']

	def create_subgraph(self, subgraphs):
		path = self.data_dir + self.params.subgraph
		if not os.path.exists(path):
			os.makedirs(path)
		for i, subgraph in subgraphs.items():
			self.write_subg(i, path, subgraph)

	def write_subg(self, id, path, subgraph):
		with open(path + 'g' + str(id), 'w') as f:
			f.write('t #\n')
			for n, _ in subgraph.items():
				f.write('v %d 0\n' % n)
			for n, info in subgraph.items():
				for nb in info['neighbors']:
					f.write('e %d %d 0\n' % (n, nb))

	def create_kernel(self):
		kernels = {}
		for k, v in json.load(open(self.data_dir + self.params.kernel)).iteritems():
			kernels[int(k)] = v
		self.num_kernel = len(kernels)
		self.num_ns = [v['v'] for k, v in sorted(kernels.iteritems())]
		self.num_es = [len(v['e']) for k, v in sorted(kernels.iteritems())]
		with open(self.data_dir + self.params.query, 'w') as f:
			for _, val in sorted(kernels.iteritems()):
				f.write('t #\n')
				for i in xrange(val['v']):
					f.write('v %d 0\n' % i)
				for e in val['e']:
					f.write('e %d %d 0\n' % (e[0], e[1]))
				for i in xrange(val['v']):
					f.write('a %d\n' % i)

	def match(self):
		sbm_data = 'SubMatch/data/'
		if not os.path.exists(sbm_data):
			os.makedirs(sbm_data)
		call('cp %s SubMatch/%s' % (self.data_dir + self.params.query, self.params.query), shell=True)
		call('rm -rf SubMatch/output/', shell=True)
		dir = self.data_dir
		for file in os.listdir(dir + self.params.subgraph):
			if file == '.DS_Store':
				continue
			os.makedirs('SubMatch/output/%s' % file)
			call('cp %s SubMatch/data/' % (dir + self.params.subgraph + file), shell=True)
			command = 'wine SubMatch/SubMatch.exe mode=2 data=SubMatch/data/%s query=SubMatch/%s maxfreq=100 stats=SubMatch/output/%s' % \
					  (file, self.params.query, file)
			call(command, shell=True)
			self.merge(file)
		call('rm SubMatch/data/g*; rm result; rm SubMatch/%s' % self.params.query, shell=True)
		call('rm subgraphs; rm -rf SubMatch/output/', shell=True)


	def read_meta(self, file):
		def read_one(f):
			line = f.readline()
			if line == '':
				return None
			line = line.rstrip().split()
			f.readline()
			f.readline()
			f.readline()
			f.readline()
			f.readline()
			return (int(line[-2]), int(line[-1]))

		num_ns, num_es = [], []
		with open(file, 'r') as f:
			while True:
				pair = read_one(f)
				if pair is None:
					break
				num_ns.append(pair[0])
				num_es.append(pair[1])
		return num_ns, num_es


	def merge(self, dir):
		meta_dir = self.data_dir + self.params.meta
		if not os.path.exists(meta_dir):
			os.makedirs(meta_dir)
		with open(meta_dir + dir, 'w') as fw:
			num_ns, num_es = self.read_meta('subgraphs')
			assert len(num_ns) == len(num_es)
			miss = 0
			for i in xrange(self.num_kernel):
				fw.write('#\t%d\t%d\t%d\n' % ((i + 1), self.num_ns[i], 2 * self.num_es[i]))
				if i - miss >= len(num_ns) or num_ns[i - miss] != self.num_ns[i] or num_es[i - miss] != self.num_es[i]:
					miss += 1
					continue
				file = 'SubMatch/output/' + dir + '/' + str(i - miss + 1)
				if not os.path.exists(file):
					continue
				with open(file, 'r') as fr:
					for line in fr:
						fw.write(line)


if __name__ == '__main__':
	prep = Prepare()
	prep.create_subgraph(prep.subgraphs)
	prep.create_kernel()
	prep.match()
