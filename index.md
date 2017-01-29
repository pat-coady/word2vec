## Overview

I've wanted to learn Google's TensorFlow for some time now. I've used Keras with TensorFlow as the back-end. Recently, I bumped into a situation where I struggled to make Keras build the neural net architecture I wanted. So it was time to program directly using the TensorFlow API. Like anything, you learn by doing.

I chose to build a simple word-embedding neural net. This seemed a good compromise that was interesting, but not so complex that I would be debugging my neural net and TensorFlow simultaneously.

## Word Vectors

A word vector is just a n-dimensional real-valued vector for each word in some dictionary (e.g. English). Word vectors for 2 similar words should be close to each other using some distance metric. For instance, you would expect "jacket" and "coat" to be close to each other. Similarly, you would expect "jacket" and "jackets" to be close.

But "jacket" is singular and "jackets" is plural, so there should be some difference in the vectors. Fine, the word with an "s" at the end is plural. But, what about "mouse" and "mice" or "goose" and "geese"? Who wants to teach a computer all of the obscure rules of English. And then repeat the task for Spanish, German, French and on. Can we learn word vectors without human-labeled data (i.e. unsupervised learning)?

## Unsupervised Learning

Learning word vectors from documents is a nice way to see the power of unsupervised learning. It can be counter-intuitive that a machine can learn anything useful without human-labeled data. I personally associated unsupervised learning with k-means clustering, and not much else. While clustering is useful, it hardly seems exciting. But unsupervised learning is exciting.

The Big Idea is to learn the structure in your data before using your precious hand-labeled examples. Here are 2 approaches:

1. Hire a bunch of people to label thousands of words in dozens of documents (e.g. plural, singular, verb, common noun, proper noun). Then set your learning algorithm loose and hope it will learn features that generalize. But, because of time contraints, your training set never encountered "Angela Merkel". Maybe the algorithm sees capital letters and figures out she is a person. But, probably not much else.
2. Train a model on **thousands** of documents and a million words. Learn patterns of words that seem interchangeable. Patterns of words that occur together. And so on. After this training, introduce a much smaller set of labeled data. When you label one word, the model will already know the similarity and relationship to dozens of other words. Now if you label "Angela Merkel" as "head of state", the model will figure out that Francois Hollande is also a head of state (having never been explicity told). And, furthermore, it will have already learned that Francois is a man's name and Angela is a woman's name. And Merkel is German and Hollande French.

## Learning Word Vectors


















With no human teacher, a neural net can learn that the words "seven" and "eight" are similar. The network also learns slightly more subtle relationships: "looking is to look as see is to \_\_\_." The algorithm learns with only the raw text of three Sir Arthur Conan Doyle Sherlock Holmes books (thanks to Project Gutenberg). The algorithm has no access to dictionaries or the Internet, only the raw ASCII text of these 3 books.
By learning word vectors with unlabeled training data, you save your precious human-labeled data for other aspects of learning. Versus "wasting" this labeled data on learning the basic structure of your problem. This approach has been applied to many applications (aside from "reading" text) with great success.
This notebook illustrates how a neural net can learn meaningful vector representations of English words. However, my primary objective was to learn TensorFlow. I built 3 python modules that support this notebook:
wordvector.py: A handy python Class (WordVector) for exploring word vectors learned by the neural network. For example:
a) Returns words that are "close" to each other based on different distance metrics
b) "Guess" the 4th word in an analogy
c) Project the vectors onto 2D space for plotting
windowmodel.py: This is where the TensorFlow graph, training and prediction routines reside. The class also contains a static method to build the training set from an integer word array.
docload.py: Load ASCII documents (with some special hooks for Project Gutenberg books). Returns the book as an numpy integer array, along with dictionaries for converting back-and-forth between integers and the English words.
