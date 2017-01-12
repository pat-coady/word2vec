"""
Unit tests for windowmodel.py

Written by Patrick Coady (pcoady@alum.mit.edu)
"""

from unittest import TestCase
import numpy as np


class TestWindowModel(TestCase):
    def test_build_training_set1(self):
        from windowmodel import WindowModel
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        x, y = WindowModel.build_training_set(a)
        self.assertEqual((5, 4), x.shape, 'incorrect data set size')
        self.assertEqual((5, 1), y.shape, 'incorrect data set size')
        self.assertTrue(np.array_equal(np.array([1, 2, 4, 5]), x[0, :]),
                        'incorrect feature value')
        self.assertTrue(np.array_equal(np.array([5, 6, 8, 9]), x[4, :]),
                        'incorrect feature value')
        self.assertTrue(np.array_equal(np.array([3, 4, 6, 7]), x[2, :]),
                        'incorrect feature value')
        self.assertEqual(3, y[0, 0], 'incorrect target value')
        self.assertEqual(7, y[4, 0], 'incorrect target value')
        self.assertEqual(5, y[2, 0], 'incorrect target value')
