import json
import os
from subprocess import call

from Predictor import Graph
from main import FLAGS


class Preprocess(object):
	def __init__(self, params=FLAGS):
		self.params = params
		self.graph = Graph(params)
		self.data_dir = 'data/' + self.params.dataset + '/'
		self.cascade = self.read_cascade(self.data_dir + self.params.data)

	def read_cascade(self, path):
		cascade = []
		with open(path, 'r') as f:
			for line in f:
				cascade.append(map(int, line.strip().split()))
		return cascade

	def create_subgraph(self, cascade):
		path = self.data_dir + self.params.subgraph
		if not os.path.exists(path):
			os.makedirs(path)
		for i, cas in enumerate(cascade):
			self.write_subg(cas, path, i)

	def write_subg(self, ns, path, id):
		ns = sorted(ns)
		es = self.graph.subgraph_es(ns)
		with open(path + 'g' + str(id), 'w') as f:
			f.write('t #\n')
			for n in ns:
				f.write('v %d 0\n' % n)
			for e in es:
				f.write('e %d %d 0\n' % (e[0], e[1]))

	def create_kernel(self):
		kernels = {}
		for k, v in json.load(open(self.data_dir + self.params.kernel)).items():
			kernels[int(k)] = v
		self.num_kernel = len(kernels)
		self.num_ns = [v['v'] for k, v in sorted(kernels.items())]
		self.num_es = [len(v['e']) for k, v in sorted(kernels.items())]
		with open(self.data_dir + self.params.query, 'w') as f:
			for _, val in sorted(kernels.items()):
				f.write('t #\n')
				for i in range(val['v']):
					f.write('v %d 0\n' % i)
				for e in val['e']:
					f.write('e %d %d 0\n' % (e[0], e[1]))
				for i in range(val['v']):
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
			real_to_fake, fake_to_real, run = self.rewrite_input('SubMatch/data/%s' % file)
			if run:
				command = 'wine SubMatch/SubMatch.exe mode=2 data=SubMatch/data/%s query=SubMatch/%s maxfreq=100 stats=SubMatch/output/%s' % \
						  (file, self.params.query, file)
				call(command, shell=True)
				self.rewrite_output('SubMatch/output/%s/' % file, fake_to_real)
			self.merge(file)
		call('rm -rf SubMatch/data/; rm result; rm SubMatch/%s' % self.params.query, shell=True)
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
			for i in range(self.num_kernel):
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


	def rewrite_input(self, file):
		real_to_fake, fake_to_real = {}, {}
		es = []
		fake_id = 0
		with open(file, 'r') as f:
			next(f)
			for line in f:
				line = line.rstrip().split()
				if line[0] == 'v':
					real_to_fake[int(line[1])] = fake_id
					fake_to_real[fake_id] = int(line[1])
					fake_id += 1
				else:
					es.append((real_to_fake[int(line[1])], real_to_fake[int(line[2])]))
		with open(file, 'w') as f:
			f.write('t #\n')
			for n in range(fake_id):
				f.write('v %d 0\n' % n)
			for e in es:
				f.write('e %d %d 0\n' % (e[0], e[1]))
		return real_to_fake, fake_to_real, len(es) > 0

	def rewrite_output(self, dir, fake_to_real):
		for file in os.listdir(dir):
			ns = []
			with open(dir + file, 'r') as f:
				for line in f:
					ns.append(map(int, line.rstrip().split()))
			with open(dir + file, 'w') as f:
				for n in ns:
					f.write('\t'.join(map(lambda x : str(fake_to_real[x]), n)) + '\n')


if __name__ == '__main__':
	preproc = Preprocess()
	preproc.create_subgraph(preproc.cascade)
	preproc.create_kernel()
	preproc.match()
