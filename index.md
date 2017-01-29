## Overview

I've wanted to learn Google's TensorFlow for some time now. I've used Keras with TensorFlow as the back-end. Recently, I bumped into a situation where I struggled to make Keras build the neural net architecture I wanted. So it was time to program directly using the TensorFlow API. Like anything, you learn by doing.

I chose to build a simple word-embedding neural net. This seemed a good compromise that was interesting, but not so complex that I would be debugging my neural net and TensorFlow simultaneously.

## Word Embedding

Learning word vectors from documents is a nice way to see the power of unsupervised learning. At first I first found it counter-intuitive that a machine could learn anything useful without human-labeled data. I associated unsupervised learning with k-means clustering, and not much else. While clustering is useful, it hardly seemed exciting.

The idea of word-embedding to learn a vector representation for each word. 










With no human teacher, a neural net can learn that the words "seven" and "eight" are similar. The network also learns slightly more subtle relationships: "looking is to look as see is to \_\_\_." The algorithm learns with only the raw text of three Sir Arthur Conan Doyle Sherlock Holmes books (thanks to Project Gutenberg). The algorithm has no access to dictionaries or the Internet, only the raw ASCII text of these 3 books.
By learning word vectors with unlabeled training data, you save your precious human-labeled data for other aspects of learning. Versus "wasting" this labeled data on learning the basic structure of your problem. This approach has been applied to many applications (aside from "reading" text) with great success.
This notebook illustrates how a neural net can learn meaningful vector representations of English words. However, my primary objective was to learn TensorFlow. I built 3 python modules that support this notebook:
wordvector.py: A handy python Class (WordVector) for exploring word vectors learned by the neural network. For example:
a) Returns words that are "close" to each other based on different distance metrics
b) "Guess" the 4th word in an analogy
c) Project the vectors onto 2D space for plotting
windowmodel.py: This is where the TensorFlow graph, training and prediction routines reside. The class also contains a static method to build the training set from an integer word array.
docload.py: Load ASCII documents (with some special hooks for Project Gutenberg books). Returns the book as an numpy integer array, along with dictionaries for converting back-and-forth between integers and the English words.
