# Overview

I have wanted to learn Google's TensorFlow for some time now. I've used Keras with TensorFlow as the back-end. Recently, I bumped into a situation where I struggled to make Keras build the neural net architecture I wanted. So it was time to program directly using the TensorFlow API. Like anything, you learn by doing.

I chose to build a simple word-embedding neural net. This seemed a good compromise that was interesting, but not so complex that I would be debugging my neural net and TensorFlow simultaneously.

# Word Vectors

A word vector is just a n-dimensional real-valued vector for each word in some dictionary (e.g. English). Word vectors for 2 similar words should be close to each other using some distance metric. For instance, you would expect "jacket" and "coat" to be close to each other. Similarly, you would expect "jacket" and "jackets" to be close.

But "jacket" is singular and "jackets" is plural, so there should be some difference in the vectors. Fine, the word with an "s" at the end is plural. But, what about "mouse" and "mice" or "goose" and "geese"? Who wants to teach a computer all of the obscure rules of English. And then repeat the task for Spanish, German, French and on. Can we learn word vectors without human-labeled data (i.e. unsupervised learning)?

# Unsupervised Learning

Learning word vectors from documents is a nice way to see the power of unsupervised learning. It can be counter-intuitive that a machine can learn anything useful without human-labeled data. Personally, I associated unsupervised learning with k-means clustering, and not much else. While clustering is useful, it hardly seems exciting. But unsupervised learning is exciting.

The Big Idea is to learn the structure in your data before using your precious hand-labeled examples. Consider these two approaches:

1. Hire a bunch of people to label thousands of words in dozens of documents - all unlabeled data. Then set your learning algorithm loose and hope it will learn features that generalize. But, because of time contraints, your training set never encountered "Angela Merkel". Maybe the algorithm sees capital letters and figures out she is a person. But, probably not much else.
2. Train a model on **hundreds** of documents and **millions** words. Learn patterns of words that seem interchangeable. Patterns of words that occur together. And so on. After this training, introduce a much smaller set of labeled data. When you label one word, the model will already know its similarity and relationship to dozens of other words. For example, if you label "Angela Merkel" as "head of state" and "Germany", the model will figure out that Francois Hollande is also a head of state, and of France (having never been explicity told).

I hope this gives you some appreciation for the power of adding an unsupervised step to a machine learning problem. This same approach applies to many domains, including speech and image recognition.

# Learning Word Vectors

There are many approaches to learning vector representations of words. The approach used here is to train a model predict the "middle" word given the N preceding and N following words. Here is a diagram of the model:

![](images/NN_diagram.png)

Here are a few key points to observe:

* The weights from the input One-Hot-Encoding (OHE) to the embedding layer are all "tied" together. This means the same weight vector is used for input word(n-2) as word(n-1), and so forth. This forces the model to learn the same representation of an input word, regardless of its input position.
* It is not practical to have a softmax with a width of the entire vocabulary. Noise Contrastive Estimation is used. Briefly, the target word and N-1 random words (drawn from a distribution matching word occurences) are used. So training is done with a N-way softmax (I've used N=64).
* There are 2 places in the model to grab learned word vectors from:
	1. The usual embedding weights: from input OHE to embedding layer
	2. The weights from the hidden layer to the softmax layer. It turns out, word vectors from here gave qualitatively more sensible results.

# Results

The model was training on 3 Sherlock Holmes books (written by Sir Arthur Conan Doyle):

* The Hound of the Baskervilles
* The Sign of the Four
* The Adventures of Sherlock Holmes

Kind thanks to Project Gutenberg.

The total vocabulary  size of three books was 11,750 words. The total number of words in the three books was ~245,000. The model was trained using stochastic gradient descent (with momentum). I got better results with SGD with momentum than with either RMSProp or Adam optimizers. The batch size was 32, with a learning rate of 0.01 and momentum of 0.9.

I constrained the word vectors to a length (dimension) of 64. Word vectors with a width of 128 had lower training and validation loss. But the results are more interesting when the model is forced to learn a more economical representation.

#### Word Similarity

Here I plug in a word and find the 8 closest words based on cosine similarity to the learned word vectors.

**"seven" : ['eight', 'five', 'ten', 'eleven', 'twelve', 'six', 'four', 'nine']**
**"laughing" : ['smiling', 'rising', 'chuckling', "'and", 'wild', 'pensively', "'well", 'yawning']**
**"mr"" : ['mrs', 'dr', "'mr", 'blind', 'earnestly', 'l', 'servant', 'st']**

I personally find that amazing. At the beginning of training, this model didn't know seven, five or ten were all numbers. The computer had never "read a book" in its life. And, now it has only read 3 Sherlock Holmes books, and it figured these relationships out.

#### Analogies

#### The Code

See the README.md file on the github page for a description of the code. I spent some time making the code readable and, hopefully, reusable. There are even some unit tests for many of the methods.















With no human teacher, a neural net can learn that the words "seven" and "eight" are similar. The network also learns slightly more subtle relationships: "looking is to look as see is to \_\_\_." The algorithm learns with only the raw text of three Sir Arthur Conan Doyle Sherlock Holmes books (thanks to Project Gutenberg). The algorithm has no access to dictionaries or the Internet, only the raw ASCII text of these 3 books.
By learning word vectors with unlabeled training data, you save your precious human-labeled data for other aspects of learning. Versus "wasting" this labeled data on learning the basic structure of your problem. This approach has been applied to many applications (aside from "reading" text) with great success.
This notebook illustrates how a neural net can learn meaningful vector representations of English words. However, my primary objective was to learn TensorFlow. I built 3 python modules that support this notebook:
wordvector.py: A handy python Class (WordVector) for exploring word vectors learned by the neural network. For example:
a) Returns words that are "close" to each other based on different distance metrics
b) "Guess" the 4th word in an analogy
c) Project the vectors onto 2D space for plotting
windowmodel.py: This is where the TensorFlow graph, training and prediction routines reside. The class also contains a static method to build the training set from an integer word array.
docload.py: Load ASCII documents (with some special hooks for Project Gutenberg books). Returns the book as an numpy integer array, along with dictionaries for converting back-and-forth between integers and the English words.

* Add t-SNE plot
* Add link to key papers
* Embed links in documents to a few key items: TensorFlow, Keras, ...