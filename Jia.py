from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''
Can run this directly
'''
import operator
import sys
import random
import os
import tensorflow as tf
import numpy as np
import pprint
from collections import deque
import metrics
import copy

random.seed(0)
np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_string("data_dir", 'cascade-datasets/twitter', "data directory.")
flags.DEFINE_integer("max_samples", 300000, "max number of samples.")
flags.DEFINE_integer("emb_dim", 64, "embedding dimension.")
flags.DEFINE_integer("disp_freq", 100, "frequency to output.")
flags.DEFINE_integer("save_freq", 10000, "frequency to save.")
flags.DEFINE_integer("test_freq", 100, "frequency to evaluate.")
flags.DEFINE_float("lr", 0.001, "initial learning rate.")
flags.DEFINE_float("positive_sample_rate", 0.5, "rate of positive samples.")
flags.DEFINE_boolean("reload_model", 0, "whether to reuse saved model.") # Note : this is for saved model
flags.DEFINE_boolean("train", 1, "whether to train model.")

FLAGS = flags.FLAGS


class Options(object):
    """options used by CDK model."""

    def __init__(self):
        # model options.
        self.emb_dim = FLAGS.emb_dim

        self.train_data = os.path.join(FLAGS.data_dir, 'train.txt')
        self.test_data = os.path.join(FLAGS.data_dir, 'test.txt')
        self.save_path = os.path.join(FLAGS.data_dir, 'embedded_ic/embedded_ic.ckpt')
        self.positive_sample_rate = FLAGS.positive_sample_rate
        self.max_samples = FLAGS.max_samples
        self.lr = FLAGS.lr
        self.disp_freq = FLAGS.disp_freq
        self.save_freq = FLAGS.save_freq
        self.test_freq = FLAGS.test_freq
        self.reload_model = FLAGS.reload_model


class Embedded_IC(object):
    """Embedded IC model."""

    def __init__(self, options, session):
        self._maxlen = 30 # max. length of cascade.
        self._options = options
        self._session = session
        self._u2idx = {}
        self._idx2u = []
        self._buildIndex()
        self._n_words = len(self._u2idx)
        self._train_cascades = self._readFromFile(options.train_data)
        self._test_cascades = self._readFromFile(options.test_data)
        self._options.train_size = len(self._train_cascades)
        self._options.test_size = len(self._test_cascades)
        self.buildGraph()

        if options.reload_model:
            self.saver.restore(session, options.save_path)

    def _getUsers(self, datafile):
        user_set = set()
        for line in open(datafile, 'rb'):
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            users = [query] + cascade.split()[::2][:self._maxlen]
            user_set.update(users)

        return user_set

    def _buildIndex(self):
        """
        compute an index of the users that appear at least once in the training and testing cascades.
        """
        opts = self._options
        user_set = self._getUsers(opts.train_data) | self._getUsers(opts.test_data)
        self._idx2u = list(user_set)
        self._u2idx = {u: i for i, u in enumerate(self._idx2u)}
        opts.user_size = len(user_set)

    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            userlist = [query] + cascade.split()[::2][:self._maxlen]
            userlist = [self._u2idx[v] for v in userlist]

            if len(userlist) > 1:
                t_cascades.append(userlist)

        return t_cascades

    def buildGraph(self):
        opts = self._options
        u = tf.placeholder(tf.int32, shape=())
        v = tf.placeholder(tf.int32, shape=())
        p_v_hat = tf.placeholder(tf.float32, shape=())
        p_uv_hat = tf.placeholder(tf.float32, shape=())

        emb_sender = tf.Variable(tf.random_uniform([opts.user_size, opts.emb_dim], -0.1, 0.1),
                                 name='emb_sender')

        emb_receiver = tf.Variable(tf.random_uniform([opts.user_size, opts.emb_dim], -0.1, 0.1),
                                   name='emb_receiver')

        u_emb = tf.nn.embedding_lookup(emb_sender, u)
        v_emb = tf.nn.embedding_lookup(emb_receiver, v)

        u_0 = tf.slice(u_emb, [0], [1])[0]
        v_0 = tf.slice(v_emb, [0], [1])[0]
        v_0_all = tf.squeeze(tf.slice(emb_receiver, [0, 0], [-1, 1]))

        u_1_n = tf.slice(u_emb, [1], [-1])
        v_1_n = tf.slice(v_emb, [1], [-1])
        v_1_n_all = tf.squeeze(tf.slice(emb_receiver, [0, 1], [-1, -1]))

        x = u_0 + v_0 + tf.reduce_sum(tf.square(u_1_n - v_1_n))
        x_all = u_0 + v_0_all + tf.reduce_sum(tf.square(v_1_n_all - u_1_n), axis=1)

        f = tf.sigmoid(-x)
        f_all = tf.sigmoid(-x_all)

        eps = 1e-8
        loss1 = -(1.0 - p_uv_hat / (p_v_hat + eps)) * tf.log(1.0 - f + eps) - (p_uv_hat / (p_v_hat + eps)) * tf.log(f + eps)
        loss2 = -tf.log(1.0 - f + eps)

        tvars = tf.trainable_variables()

        grads1 = tf.gradients(loss1, tvars)
        grads2 = tf.gradients(loss2, tvars)
        grads1, _ = tf.clip_by_global_norm(grads1, clip_norm=5.)
        grads2, _ = tf.clip_by_global_norm(grads2, clip_norm=5.)

        train1 = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(grads1, tvars))
        train2 = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(grads2, tvars))

        self.u = u
        self.v = v
        self.p_uv_hat = p_uv_hat
        self.p_v_hat = p_v_hat
        self.emb_sender = emb_sender
        self.emb_receiver = emb_receiver
        self.p_uv = f
        self.p_u_all = f_all
        self.loss1 = loss1
        self.loss2 = loss2
        self.train1 = train1
        self.train2 = train2

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def SampleCascade(self):
        """sample a cascade randomly."""
        opts = self._options
        return random.randint(0, opts.train_size - 1)

    def SampleV(self, cascadeId):
        """
        sample a user V, which can not be the initial user of the given cascade.
        with `positive_sample_rate` probability V is in the given cascade (positive sample).
        """
        opts = self._options
        c = self._train_cascades[cascadeId]

        # if True:
        if random.random() > opts.positive_sample_rate:
            while True:
                idx = random.randint(0, opts.user_size - 1)
                if idx != c[0]:
                    break
        else:
            i = random.randint(1, len(c) - 1)
            idx = c[i]

        v_in_cascade = idx in set(self._train_cascades[cascadeId])
        return v_in_cascade, idx

    def SampleU(self, cascadeId, vId):
        """sample users u given sampled cascade and user v."""
        ulist = []
        for user in self._train_cascades[cascadeId]:
            if user == vId:
                break
            ulist.append(user)

        return ulist

    def computePv(self, v, ul):
        '''computes \hat{P}_v'''
        pv = 1.0
        assert len(ul) > 0, (v, ul)
        for u in ul:
            feed_dict = {self.u: u, self.v: v}
            p_uv = self._session.run(self.p_uv, feed_dict=feed_dict)
            pv = pv * (1.0 - p_uv)
        p_v = 1.0 - pv
        return p_v


    def train(self):
        """train the model."""
        opts = self._options
        n_samples = 0
        for _ in xrange(opts.max_samples):
            cascade_id = self.SampleCascade()
            v_in_cascade, v_id = self.SampleV(cascade_id)
            u_id_list = self.SampleU(cascade_id, v_id)
            if v_in_cascade:
                p_v_hat = self.computePv(v_id, u_id_list)
                for u in u_id_list:
                    p_uv_hat = self._session.run(self.p_uv,
                                                 feed_dict={self.u: u, self.v: v_id})
                    loss, _ = self._session.run([self.loss1, self.train1],
                                                feed_dict={self.u: u,
                                                           self.v: v_id,
                                                           self.p_uv_hat:
                                                           p_uv_hat,
                                                           self.p_v_hat: p_v_hat})
            else:
                for u in u_id_list:
                    loss, _ = self._session.run([self.loss2, self.train2],
                                                feed_dict={self.u: u, self.v: v_id})

            n_samples += 1
            #print(n_samples)
            if n_samples % opts.disp_freq == 0:
                print('step %d, loss=%f' % (n_samples, loss))
            if n_samples % opts.save_freq == 0:
                self.saver.save(self._session, opts.save_path)
            if n_samples % opts.test_freq == 0:
                # your evaluation function here.
                print ("")

    def simulateOneStep(self,P, seedSet, K, max_iter = 1000):
        opts = self._options
        activationProb = {}
        for i in range(0, opts.user_size):
            activationProb[i] = 0.0
        
        for iterNum in range(0,max_iter):
            activeSet = {}
            for v in seedSet:
                rand_vector = np.random.rand(opts.user_size)
                rand_vector = P[v] - rand_vector
                a = rand_vector.clip(min=0).nonzero()[0].tolist() # "activated" nodes.
                for x in a:
                    if x not in seedSet:
                        activeSet[x] = 1
            for x in activeSet:
                activationProb[x] += 1.0

        for i in range(0, len(activationProb)):
            activationProb[i] /= max_iter
        activatedUsers = sorted(activationProb.items(), key = operator.itemgetter(1), reverse = True)[0:K]
        return  map(lambda x: x[0], activatedUsers)
    # This does evalution by just ranking the probabilities - top-k MAP, etc.
    # No monte-carlo simulations. 

    def evalOneStep(self, K):
        opts = self._options
        # Compute all pairwise probabilities.
        P = np.zeros(shape = (opts.user_size, opts.user_size))
        for i in range(0, opts.user_size):
            feed_dict = {self.u: i}
            t = self._session.run(self.p_u_all, feed_dict=feed_dict)
            P[i] = t
        HITS = 0.0
        MAP = 0.0
        for cascade in self._test_cascades:
            print ("Cascade",cascade)
            avgPrecision = 0.0
            avgHITS = 0.0
            for j in range(1, len(cascade)):
                seedSet = cascade[0:j]
                nextUser = cascade[j]
                predictedUsers = self.simulateOneStep(P, seedSet, K, 100)
                if nextUser in predictedUsers:
                    avgHITS += 1.0
                    avgPrecision += 1.0/(predictedUsers.index(nextUser)+1)
                
            avgPrecision /= (len(cascade)-1)
            avgHITS /= (len(cascade)-1)
            print("Avg. prec. ", avgPrecision)
            print("Avg. hits ", avgHITS)
            MAP += avgPrecision
            HITS += avgHITS
        MAP /= len(self._test_cascades)
        HITS /= len(self._test_cascades)
        print ("MAP", MAP)
        print ("HITS", HITS)


    def eval(self, K):
        opts = self._options
        P = np.zeros(shape = (opts.user_size, opts.user_size))
        for i in range(0, opts.user_size):
            feed_dict = {self.u: i}
            t = self._session.run(self.p_u_all, feed_dict=feed_dict)
            P[i] = t

        # Return MAP on test set for input K- top K
        MAP = 0.0
        for cascade in self._test_cascades:
            print ("Cascade",cascade)
            # Create multiple seed sets and try to predict the next node. 
            avg_precision = 0.0
            for j in range(1, len(cascade)):
                seedSet = cascade[0:j]
                nextUser = cascade[j]
                # Check if nextUser is predicted correctly by the model.
                pred_users = self.simulateModel(P, seedSet, K,10)
                precision = 0.0
                for i in range(0,K):
                    if pred_users[i] == nextUser:
                        precision = 1
                        break
                avg_precision += precision
            avg_precision /= (len(cascade)-1)
            print ("Avg. hits = ", avg_precision)
            MAP += avg_precision
        MAP /= len(self._test_cascades)
        print ("MAP = ", MAP)        


def main(_):
    options = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = Embedded_IC(options, session)
        if FLAGS.train:
            model.train()
        else:
            model.evalOneStep(10)


if __name__ == "__main__":
    tf.app.run()
