"""
Learning Word Vectors: Window Approach
Written by Patrick Coady (pcoady@alum.mit.edu)

This window-based model learns word vectors (or "embedding vectors")
based on neighboring words. The word vectors are learned as the models
tries to predict the middle word based on the 2 preceding and 2
following words.

The WindowModel object contains a TensorFlow graph. Class methods train
and evaluate the TensorFlow graph. The model returns word vectors and can
also predicts the middle word based on 2 prior and 2 following words.

The WindowModel class has a static helper method to convert a
numpy array of integers (representing words in a document) into a
properly formatted training example.
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class WindowModel(object):
    def __init__(self, graph_params):
        """
        Takes dictionary of parameters and builds TensorFlow graph.
        :param graph_params: Expected dictionary entries:
            'batch_size': (int) stochastic gradient descent (SGD) batch size
            'vocab_size': (int) number of words in dictionary. words are
                represented as integers, so vocab_size = max(word)+1 (first
                word = 0).
            'embed_size': (int) length of word (embedding) vector
            'hid_size': (int) size of hidden layer
            'neg_samples': (int) number of samples to use for noise contrastive
                estimation. 64 is a reasonable choice.
            'learn_rate': (float) SGD learning rate
        """
        self.graph = tf.Graph()
        self.g_params = graph_params.copy()
        self.g_ops_tensors = dict()
        self._build_tfgraph()

    def _build_tfgraph(self):
        """ Builds TensorFlow graph. Should only be called by __init__() """
        with self.graph.as_default():
            g_ops_tensors = self.g_ops_tensors  # shorten var names
            g_params = self.g_params            # shorten var names
            # model inputs: x = input, target = training label
            g_ops_tensors['x'] = tf.placeholder(
                tf.int32, shape=[g_params['batch_size'], 4])
            g_ops_tensors['target'] = tf.placeholder(
                tf.int32, shape=[g_params['batch_size'], 1])
            w_m2, w_m1, w_p1, w_p2 = tf.unstack(
                g_ops_tensors['x'], axis=1)

            # embedding layer
            g_ops_tensors['embed_weights'] = tf.Variable(
                tf.random_uniform([g_params['vocab_size'],
                                   g_params['embed_size']],
                                  -1.0, 1.0))  # TODO: Is this reasonable initialization?
            # shared embedding weights for all 4 input words
            embed_m2 = tf.nn.embedding_lookup(g_ops_tensors['embed_weights'], w_m2)
            embed_m1 = tf.nn.embedding_lookup(g_ops_tensors['embed_weights'], w_m1)
            embed_p1 = tf.nn.embedding_lookup(g_ops_tensors['embed_weights'], w_p1)
            embed_p2 = tf.nn.embedding_lookup(g_ops_tensors['embed_weights'], w_p2)
            # concatenate word vectors for all 4 input words
            embed_stack = tf.concat(concat_dim=1,
                                    values=[embed_m2, embed_m1, embed_p1, embed_p2])

            # hidden layer
            hid_weights = tf.Variable(  # TODO: Is this reasonable random initialization?
                tf.random_normal([g_params['embed_size'] * 4,
                                  g_params['hid_size']],
                                 stddev=1.0/(g_params['embed_size'] * 4)**0.5))
            hid_bias = tf.Variable(tf.zeros([g_params['hid_size']]))
            hid_out = tf.nn.tanh(tf.matmul(embed_stack, hid_weights) + hid_bias)

            # output layer
            g_ops_tensors['nce_weights'] = tf.Variable(  # TODO: Why truncated normal?
                tf.truncated_normal([g_params['vocab_size'],
                                     g_params['hid_size']],
                                    stddev=1.0/g_params['hid_size']**0.5))
            nce_bias = tf.Variable(tf.zeros([g_params['vocab_size']]))

            # loss function - noise contrastive estimation
            g_ops_tensors['loss'] = tf.reduce_mean(
                tf.nn.nce_loss(g_ops_tensors['nce_weights'], nce_bias,
                               inputs=hid_out, labels=g_ops_tensors['target'],
                               num_sampled=g_params['neg_samples'],
                               num_classes=g_params['vocab_size'],
                               num_true=1, remove_accidental_hits=True))
            # TODO: send actual word prob. dist. to nce_loss function

            # optimizer
            g_ops_tensors['optimizer'] = tf.train.RMSPropOptimizer(
                g_params['learn_rate']).minimize(g_ops_tensors['loss'])

            # variable init
            self.g_ops_tensors['initializer'] = tf.global_variables_initializer()

        return None

    def eval_loss(self, x_val, y_val, session):
        """
        Calculate and return evaluation loss on a test set.
        :param x_val: np.array(dtype=np.int32, shape=(N,4)) features
        :param y_val: np.array of shape (vocab_size, hid_size): targets
        :param session: TensorFlow session with trained graph
        :return: (float) average loss (noise contrastive estimation loss)
        """
        g_ops_tensors = self.g_ops_tensors  # shorten var names
        g_params = self.g_params            # shorten var names
        num_samples = x_val.shape[0]
        num_batches = num_samples // g_params['batch_size']
        avg_loss = 0
        for batch in range(num_batches):
            bot_idx = batch * g_params['batch_size']
            top_idx = bot_idx + g_params['batch_size']
            feed_dict = {g_ops_tensors['x']: x_val[bot_idx:top_idx, :],
                         g_ops_tensors['target']: y_val[bot_idx:top_idx, :]}
            loss = session.run([g_ops_tensors['loss']], feed_dict=feed_dict)
            avg_loss += loss[0]

        return avg_loss/num_batches

    # TODO: over-training near end. implement drop-out.
    # TODO: tune hyper-parameters
    def train(self, x, y, x_val=None, y_val=None, epochs=1):
        """
        Train TensorFlow graph. In-sample and validation error reported (stdout) after
        each training epoch. 2 word vectors returned:
            1) Input embedding vector
            2) Hidden layer to output word vector
        :param x: np.array(dtype=np.int32, shape=(N,4)) training features
        :param y: np.array(dtype=np.int32, shape=(N,1)) training targets
        :param x_val: validation data set, features (same format as training data)
        :param y_val: validation data set, targets (same format as training data)
        :param epochs: (int) number of training epochs
        :return: 2-tuple of word embedding matrices:
            1. np.array of shape (vocab_size, embed_size): input embedding
            2. np.array of shape (vocab_size, hid_size): hid to output word vector
        """
        g_ops_tensors = self.g_ops_tensors  # shorten var names
        g_params = self.g_params            # shorten var names
        num_batches = len(x) // g_params['batch_size']
        avg_loss, avg_loss_count, batch_count = (0, 0, 0)
        with tf.Session(graph=self.graph) as session:
            session.run(g_ops_tensors['initializer'])   # initialize graph
            for epoch in range(epochs):
                print('epoch {}: '.format(epoch + 1), end='')
                avg_loss, avg_loss_count = (0, 0)
                x, y = shuffle(x, y)                    # shuffle data prior to each epoch
                for batch in range(num_batches):
                    bot_idx = batch * g_params['batch_size']
                    top_idx = bot_idx + g_params['batch_size']
                    feed_dict = {g_ops_tensors['x']: x[bot_idx:top_idx, :],
                                 g_ops_tensors['target']: y[bot_idx:top_idx, :]}
                    _, loss = session.run([g_ops_tensors['optimizer'],
                                           g_ops_tensors['loss']],
                                          feed_dict=feed_dict)
                    avg_loss += loss
                    avg_loss_count += 1
                    batch_count += 1
                print('total batches = {}. Ein = {:.2f}, Eout = {:.2f}'.format(
                    batch_count, avg_loss / avg_loss_count,
                    self.eval_loss(x_val, y_val, session)))
            embed_weights = g_ops_tensors['embed_weights'].eval()
            hid_to_out_weights = g_ops_tensors['nce_weights'].eval()

        return embed_weights, hid_to_out_weights

    @staticmethod
    def build_training_set(word_array):
        """
        Build training set for learning word vectors based on 2 neighboring
        words on each side of a target word. For example, for 'the cat sat on mat'.
        'sat' is the target. 'the', 'cat', 'on' and 'mat' are the training features.
        The data is actually passed to this method as an array of integers, with
        integers representing words.
        :param word_array: Array of integers representing words in a document. The
            array order should match the order in the document (i.e. word_array[0] is
            the first word in the document, word_array[1] is the 2nd word, ...)
        :return: 2-tuple (features, target):
            1. np.array(dtype=np.int32, shape=(N, 4): features (i.e. neighbor words)
            2. np.array(dtype=np.int32, shape=(N, 1): targets (i.e. middle word)
        """
        num_words = len(word_array)
        x = np.zeros((num_words-4, 4), dtype=np.int32)
        y = np.zeros((num_words-4, 1), dtype=np.int32)
        shift = np.array([-2, -1, 1, 2], dtype=np.int32)
        for idx in range(2, num_words-2):
            y[idx-2, 0] = word_array[idx]
            x[idx-2, :] = word_array[idx+shift]

        return x, y

# TODO: add cross-check graph calculation (numpy-based calculation)
# TODO: add saver so trained graph can be reloaded
# TODO: add predict_word() method
# TODO: return training/val loss after training
# TODO: experiment with swapping in predicted words on unseen document