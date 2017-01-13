"""
Unit tests for docload.py

Written by Patrick Coady (pcoady@alum.mit.edu)
"""

from unittest import TestCase
import collections


class TestDocload(TestCase):
    def test_load1(self):
        from docload import load_books
        filenames = ['test1.txt']
        word_count, word_list, num_lines, num_words = load_books(filenames, gutenberg=False)
        self.assertEqual(7, len(word_count), 'incorrect counter length')
        self.assertEqual(2, word_count['the'], 'incorrect word count')
        self.assertEqual(3, word_count['brown'], 'incorrect word count')
        self.assertEqual(1, word_count['quick'], 'incorrect word count')
        self.assertEqual(11, num_words, 'wrong number of words')
        self.assertEqual(1, num_lines, 'wrong number of lines')

    def test_load2(self):
        from docload import load_books
        filenames = ['test2.txt']
        word_count, word_list, num_lines, num_words = load_books(filenames, gutenberg=True)
        self.assertEqual(7, len(word_count), 'incorrect counter length')
        self.assertEqual(2, word_count['the'], 'incorrect word count')
        self.assertEqual(3, word_count['brown'], 'incorrect word count')
        self.assertEqual(1, word_count['quick'], 'incorrect word count')
        self.assertEqual(11, num_words, 'wrong number of words')
        self.assertEqual(4, num_lines, 'wrong number of lines')

    def test_load3(self):
        from docload import load_books
        filenames = ['test2.txt', 'test2.txt']
        word_count, word_list, num_lines, num_words = load_books(filenames, gutenberg=True)
        self.assertEqual(7, len(word_count), 'incorrect counter length')
        self.assertEqual(4, word_count['the'], 'incorrect word count')
        self.assertEqual(6, word_count['brown'], 'incorrect word count')
        self.assertEqual(2, word_count['quick'], 'incorrect word count')
        self.assertEqual(22, num_words, 'wrong number of words')
        self.assertEqual(8, num_lines, 'wrong number of lines')

    def test_build_dict(self):
        from docload import build_dict
        word_list = ['the', 'quick', 'brown', 'fox', 'jumped', 'quick', 'over', 'the',
                     'the', 'the', 'lazy', 'quick', 'dog']
        word_counter = collections.Counter(word_list)
        dictionary = build_dict(word_counter, 3)
        self.assertEqual(0, dictionary['the'], 'most frequent word not mapped to 0')
        self.assertEqual(1, dictionary['quick'], '2nd most frequent word not mapped to 1')
        self.assertEqual(3, len(dictionary), 'incorrect dictionary length')

    def test_doc2num(self):
        from docload import doc2num
        word_list = ['the', 'quick', 'brown', 'fox', 'jumped', 'quick', 'over', 'the',
                     'the', 'the', 'lazy', 'quick', 'dog']
        dictionary = {'the': 0, 'quick': 1, 'brown': 2, 'fox': 3}
        word_array = doc2num(word_list, dictionary)
        self.assertEqual(0, word_array[0], 'incorrect word to num mapping')
        self.assertEqual(1, word_array[1], 'incorrect word to num mapping')
        self.assertEqual(3, word_array[3], 'incorrect word to num mapping')
        self.assertEqual(4, word_array[12], 'incorrect word to num mapping')
        self.assertEqual(1, word_array[11], 'incorrect word to num mapping')
