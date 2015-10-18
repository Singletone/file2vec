import theano
import theano.tensor as T
import parameters
import log
import numpy as np
from matplotlib import pyplot as plt


floatX = theano.config.floatX
empty = lambda *shape: np.empty(shape, dtype='int32')
rnd1 = lambda d0: np.random.rand(d0).astype(dtype=floatX)
rnd2 = lambda d0, d1: np.random.rand(d0, d1).astype(dtype=floatX)


def train(fileIndexMap, wordIndexMap, wordEmbeddings, contexts):
    contextsCount = len(contexts)

    filesCount = len(fileIndexMap)
    fileEmbeddingSize = 100
    wordsCount = len(wordIndexMap)
    wordEmbeddingSize = wordEmbeddings.shape[1]
    windowSize = contexts.shape[1] - 1
    contextSize = windowSize - 1

    fileEmbeddings = theano.shared(rnd2(filesCount, fileEmbeddingSize), 'fileEmbeddings', borrow=True)
    wordEmbeddings = theano.shared(wordEmbeddings, 'wordEmbeddings', borrow=True)
    weights = theano.shared(rnd2(contextSize * wordEmbeddingSize, wordsCount), 'weights', borrow=True)
    biases = theano.shared(rnd1(wordsCount), 'bias', borrow=True)

    fi = contexts[:,0:1]
    wi = contexts[:,1:-1]
    ti = contexts[:,-1:]
    ti = ti.reshape(ti.shape[0])

    fileIndices = theano.shared(fi, borrow=True)
    wordIndices = theano.shared(wi, borrow=True)
    targetIndices = theano.shared(ti, borrow=True)

    words = wordEmbeddings[wordIndices]
    wordFeatures = T.reshape(words, (words.shape[0], words.shape[1] * words.shape[2]))
    probabilities = T.nnet.softmax(T.dot(wordFeatures, weights) + biases)

    parameters = [weights, biases]

    l1Coefficient = T.scalar('l1Coefficient', dtype=floatX)
    l2Coefficient = T.scalar('l2Coefficient', dtype=floatX)

    l1Regularization = l1Coefficient * sum([abs(p).sum() for p in parameters])
    l2Regularization = l2Coefficient * sum([(p ** 2).sum() for p in parameters])

    cost = -T.mean(T.log(probabilities)[T.arange(targetIndices.shape[0]), targetIndices]) + \
           l1Regularization + l2Regularization

    learningRate = T.scalar('learningRate', dtype=floatX)

    gradients = [T.grad(cost, wrt=p) for p in parameters]
    updates = [(p, p - learningRate * g) for p, g in zip(parameters, gradients)]

    trainModel = theano.function(
        inputs=[learningRate, l1Coefficient, l2Coefficient],
        outputs=cost,
        updates=updates
    )

    errors = []
    epochs = 300
    for epoch in xrange(0, epochs):
        error = trainModel(0.5, 0.006, 0.001)
        errors.append(error)

        log.progress('Training model: {0:.3f}%. Error: {1:.3f}.', epoch + 1, epochs, float(error))

    plt.grid()
    plt.scatter(np.arange(0, epochs), errors)
    plt.show()


def main():
    fileIndexMap = parameters.loadIndexMap('../data/Drosophila/Parameters/file_index_map.bin')
    wordIndexMap = parameters.loadIndexMap('../data/Drosophila/Parameters/word_index_map.bin')
    indexWordMap = parameters.loadIndexMap('../data/Drosophila/Parameters/word_index_map.bin', inverse=True)
    wordEmbeddings = parameters.loadEmbeddings('../data/Drosophila/Parameters/word_embeddings.bin')

    contextProvider = parameters.IndexContextProvider('../data/Drosophila/Processed/contexts.bin')
    contexts = contextProvider[:]

    log.info('Contexts loading complete. {0} contexts loaded {1} words each.', len(contexts), contextProvider.contextSize)

    train(fileIndexMap, wordIndexMap, wordEmbeddings, contexts)


if __name__ == '__main__':
    main()
