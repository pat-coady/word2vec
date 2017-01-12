"""
Unit tests for wordvector.py

Written by Patrick Coady (pcoady@alum.mit.edu)
"""

from unittest import TestCase
import numpy as np


class TestWordVector(TestCase):

    def n_closest(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 1.01],
                                 [2.0, 2.0],
                                 [2.0, 2.1],
                                 [1.0, 0.0],
                                 [0, 1.01],
                                 [-1.0, 0.0]])
        word_embedding = WordVector(embed_matrix, dictionary)
        nc_list = word_embedding.n_closest('quick', 3, metric='euclidean')
        self.assertEqual(['quick', 'brown', 'the'], nc_list,
                         'wrong n-closest words returned')
        nc_list = word_embedding.n_closest('quick', 2, metric='cosine')
        self.assertEqual(['the', 'fox'], nc_list,
                         'wrong n-closest words returned')

    def test_words_in_range(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.6, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.7, 0.1, 1.1, 1.1, 1.1, 0.1]])
        word_embedding = WordVector(embed_matrix, dictionary)
        range_list = word_embedding.words_in_range(3, 6)
        self.assertEqual(['fox', 'jumped', 'over'], range_list, 'wrong most common words returned')
        range_list = word_embedding.words_in_range(0, 2)
        self.assertEqual(['the', 'quick'], range_list, 'wrong most common words returned')

    def test_most_common(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.6, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.7, 0.1, 1.1, 1.1, 1.1, 0.1]])
        word_embedding = WordVector(embed_matrix, dictionary)
        mc_list = word_embedding.most_common(3)
        self.assertEqual(['the', 'quick', 'brown'], mc_list, 'wrong most common words returned')
        mc_list = word_embedding.most_common(1)
        self.assertEqual(['the'], mc_list, 'wrong most common words returned')

    def test_analogy(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.6, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.7, 0.1, 1.1, 1.1, 1.1, 0.1]])
        word_embedding = WordVector(embed_matrix, dictionary)
        d = word_embedding.analogy('the', 'fox', 'quick', num=2, metric='euclidean')
        self.assertEqual(2, len(d), 'wrong number of analogies returned')
        self.assertEqual('jumped', d[0], 'wrong most likely analogy returned')
        self.assertEqual('over', d[1], 'wrong 2nd most likely analogy returned')

    def test_project_2D_1(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1],
                                 [1.0, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1]])
        word_embedding = WordVector(embed_matrix, dictionary)
        proj, words = word_embedding.project_2d(1, 5)
        self.assertEqual((4, 2), proj.shape, 'incorrect projection array size returned')
        self.assertEqual('quick', words[0], 'incorrect word at index 0')
        self.assertEqual('jumped', words[3], 'incorrect word at index 3')

    def test_project_2D_2(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1],
                                 [1.0, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1]])
        word_embedding = WordVector(embed_matrix, dictionary)
        proj, words = word_embedding.project_2d(0, 6)
        self.assertEqual((6, 2), proj.shape, 'incorrect projection array size returned')
        self.assertEqual('the', words[0], 'incorrect word at index 0')
        self.assertEqual('fox', words[3], 'incorrect word at index 3')

    def test_num_words(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1],
                                 [1.0, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1]])
        word_embedding = WordVector(embed_matrix, dictionary)
        self.assertEqual(6,
                         word_embedding.num_words(),
                         'incorrect number of words')

    def test_get_vector_by_name(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1],
                                 [1.0, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1]])
        word_embedding = WordVector(embed_matrix, dictionary)
        self.assertTrue(np.sum(np.abs(np.array([1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1])
                        - word_embedding.get_vector_by_name('fox'))) < 0.1,
                        'incorrest closest indices')
        self.assertTrue(np.sum(np.abs(np.array([1.0, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1])
                        - word_embedding.get_vector_by_name('over'))) < 0.1,
                        'incorrest closest indices')
        self.assertTrue(np.sum(np.abs(np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                        - word_embedding.get_vector_by_name('the'))) < 0.1,
                        'incorrest closest indices')

    def test_get_vector_by_num(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1],
                                 [1.0, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1],
                                 [1.0, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1]])
        word_embedding = WordVector(embed_matrix, dictionary)
        self.assertTrue(np.sum(np.abs(np.array([1.0, 0.1, 0.1, 1.1, 1.1, 1.1, 0.1])
                        - word_embedding.get_vector_by_num(3))) < 0.1,
                        'incorrest closest indices')
        self.assertTrue(np.sum(np.abs(np.array([1.0, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1])
                        - word_embedding.get_vector_by_num(5))) < 0.1,
                        'incorrest closest indices')
        self.assertTrue(np.sum(np.abs(np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                        - word_embedding.get_vector_by_num(0))) < 0.1,
                        'incorrest closest indices')

    def test_gets(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 1.01],
                                 [2.0, 2.0],
                                 [2.0, 2.1],
                                 [1.0, 0.0],
                                 [0, 1.01],
                                 [-1.0, 0.0]])
        word_embedding = WordVector(embed_matrix, dictionary)
        d = word_embedding.get_dict()
        dr = word_embedding.get_reverse_dict()
        em = word_embedding.get_embed()
        d.pop('the')  # mutate, check that copies were returned
        dr.pop(1)
        em[0, 0] = 10
        d = word_embedding.get_dict()
        dr = word_embedding.get_reverse_dict()
        em = word_embedding.get_embed()
        self.assertEqual(6, len(d), 'wrong dictionary length')
        self.assertEqual(6, len(dr), 'wrong dictionary length')
        self.assertEqual(1.0, em[0, 0], 'wrong value in embed matrix')
        self.assertEqual(3, d['fox'], 'wrong value from dictionary')
        self.assertEqual('jumped', dr[4], 'wrong value from reverse dictionary')

    def test_closest_row_indices(self):
        from wordvector import WordVector
        dictionary = {'the': 0,
                      'quick': 1,
                      'brown': 2,
                      'fox': 3,
                      'jumped': 4,
                      'over': 5}
        embed_matrix = np.array([[1.0, 1.01],
                                 [2.0, 2.0],
                                 [2.0, 2.1],
                                 [1.0, 0.0],
                                 [0, 1.01],
                                 [-1.0, 0.0]])
        word_embedding = WordVector(embed_matrix, dictionary)
        dist_list = word_embedding.closest_row_indices(np.array([[2.0, 2.0]]), 3, 'euclidean')
        self.assertTrue(np.sum(np.abs(np.array([1, 2, 0]) - dist_list)) < 0.1,
                        'incorrest closest indices')
        dist_list = word_embedding.closest_row_indices(np.array([[2.0, 2.0]]), 3, 'cosine')
        self.assertTrue(np.sum(np.abs(np.array([1, 0, 2]) - dist_list)) < 0.1,
                        'incorrest closest indices')
        dist_list = word_embedding.closest_row_indices(np.array([[1.0, 1.0]]), 6, 'euclidean')
        self.assertTrue(np.sum(np.abs(np.array([0, 3, 4, 1, 2, 5]) - dist_list)) < 0.1,
                        'incorrest closest indices')
