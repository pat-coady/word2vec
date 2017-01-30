# Overview

See the [GitHub Project Page](https://pat-coady.github.io/word2vec/) for a high-level overview of the project.

### Python Modules

This repository contains python modules to learn word vectors from raw text documents using TensorFlow. There are 4 primary modules:

1. **docload.py**: Loads and processes raw text documents in preparation for model training. Has a few basic "hooks" to make loading Project Gutenberg books easy. Documents are represented as integer numpy arrays.
2. **windowmodel.py**: Contains the TensorFlow graph and methods to train the model, return word vectors and make predictions. Initial call returns WindowModel object. Also contains static method to take integer numpy array and format for training.
3. **wordvector.py**: Explore word vectors returned by WindowModel.train(). Finds closest words based on a variety of distance metrics. Has method to predict analogies (i.e. A is to B as C is to D). Also includes routine to project word vectors to 2D space using t-SNE.
4. **plot\_util.py**: Only 1 plot utility at this time: plot learning curves from training.

### iPython Notebooks

1. **sherlock.ipynb**: Uses above modules to load 3 *Sherlock Holmes* books, train the neural net and do some basic exploration of the results.
2. **tune\_\*.ipynb**: Hyper-parameter tuning for sherlock.ipynb model. Explore different layer sizes, learning rates, optimizers and weight initialization.
3. **word\_frequency.ipynb**: Plot word frequencies from 3 Sherlock Holmes books and overlay log-uniform distribution. Noise contrastive estimation routine (tf.nn.nce\_loss) in Tensorflow assume log-uniform word frequency distribution.

### word2vec/test

Unit tests for Python modules.
