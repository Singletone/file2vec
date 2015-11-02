# text2vec
Text vector model is supposed to train text vector representations 
described in https://cs.stanford.edu/~quocle/paragraph_vector.pdf
It runs [theano](http://deeplearning.net/software/theano/) underhood and 
can be launched on both CPU and GPU.

##Modules

All functionality is splitted into evaluation, infrustructure, injection, libs,
processing, training modules and controllers that represent certain stages
of data flow.

###Evaluation
Reading and writing training metrics, validation routines and visualization tools.

###Infrustructure
Some general wrappers that simplify access to file system. Logging.

###Injection
Abstract layers for different datasets.

###Libs
External tools (e.g. t-SNE).

###Processing
Tool set for turning raw data into a training format (creation of word/index maps, reading
word contexts/windows, negative sampling and so on).

###Training
Finally, training module provides access to text vector training functionality.

###Controllers
Implement different stages of data processing. They can be launched
either from command line or from python. When launched from a command line
each controller acts more like a standalone tool that will save work results to HDD. 
But these controllers also provide a bunch of tools to process data in-memory.

##Data processing stages
Generally data goes through the following pipeline:
Injection → Processing → Training

###Extraction
On this stage data may be collected from a number of datasets using dataset connectors.
Each connector is an iterator that provides access to the name/text pairs stored in a
dataset. No text transformations are applied on this stage except only dataset specific ones.
Thus, for instance, when injecting Wikipedia dataset all gzipped wiki dumps are unpacked into
a set of name/text pairs with all wiki markup specific noise removed.
Currently, connectors for the following datasets are available:
- [Wikipedia](http://kopiwiki.dsd.sztaki.hu/);
- [Rotten Tomatos](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews);
- [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/);
- plain text files.

###Processing
The main goal of processing is to subsample the most frequent words, prune least unique ones,
build word contexts and append them with negative samples.

###Training
Text vector training. May be launched in two modes - with weights fixed or not. When weights are 
fixed training module acts as a text vector inferer. Training module does not train word vector
representations. Instead it relies on those word vectors that are trained by word2vec tool.