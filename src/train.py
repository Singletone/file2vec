import theano
import theano.tensor as T
import parameters
import log
import numpy as np
import vectors
from matplotlib import pyplot as plt


floatX = theano.config.floatX
empty = lambda *shape: np.empty(shape, dtype='int32')
rnd1 = lambda d0: np.random.rand(d0).astype(dtype=floatX)
rnd2 = lambda d0, d1: np.random.rand(d0, d1).astype(dtype=floatX)


def train(fileIndexMap, wordIndexMap, wordEmbeddings, contexts):
    contextsCount = len(contexts)

    filesCount = len(fileIndexMap)
    fileEmbeddingSize = 200
    wordsCount = len(wordIndexMap)
    wordEmbeddingSize = wordEmbeddings.shape[1]
    windowSize = contexts.shape[1] - 1
    contextSize = windowSize - 1

    fileEmbeddings = theano.shared(rnd2(filesCount, fileEmbeddingSize), 'fileEmbeddings', borrow=True)
    wordEmbeddings = theano.shared(wordEmbeddings, 'wordEmbeddings', borrow=True)
    weights = theano.shared(rnd2(fileEmbeddingSize + contextSize * wordEmbeddingSize, wordsCount), 'weights', borrow=True)
    biases = theano.shared(rnd1(wordsCount), 'bias', borrow=True)

    ctxs = T.imatrix('contexts')
    fileIndices = ctxs[:,0:1]
    wordIndices = ctxs[:,1:-1]
    targetIndices = T.reshape(ctxs[:,-1:], (ctxs.shape[0],))

    files = fileEmbeddings[fileIndices]
    fileFeatures = T.flatten(files, outdim=2)
    words = wordEmbeddings[wordIndices]
    wordFeatures = T.flatten(words, outdim=2)
    features = T.concatenate([fileFeatures, wordFeatures], axis=1)

    probabilities = T.nnet.softmax(T.dot(features, weights) + biases)

    parameters = [fileEmbeddings, weights, biases]

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
        inputs=[ctxs, learningRate, l1Coefficient, l2Coefficient],
        outputs=cost,
        updates=updates
    )

    errors = []
    epochs = 1000
    for epoch in xrange(0, epochs):
        error = trainModel(contexts, 0.5, 0.006, 0.001)
        errors.append(error)

        biochemistry = '../data/Drosophila/Prepared/drosophila/0_biochemistry.txt'
        biology = '../data/Drosophila/Prepared/drosophila/0_biology.txt'
        galaxy = '../data/Drosophila/Prepared/drosophila/1_galaxy.txt'
        star = '../data/Drosophila/Prepared/drosophila/1_star.txt'

        biochemistryIndex = fileIndexMap[biochemistry]
        biologyIndex = fileIndexMap[biology]
        galaxyIndex = fileIndexMap[galaxy]
        starIndex = fileIndexMap[star]

        fe = fileEmbeddings.get_value()

        biochemistryVector = fe[biochemistryIndex]
        biologyVector = fe[biologyIndex]
        galaxyVector = fe[galaxyIndex]
        starVector = fe[starIndex]

        log.progress('Training model: {0:.3f}%. Error: {1:.3f}. Biochemistry/Biology={2:.3f}. Galaxy/Star={3:.3f}. Biology/Galaxy={4:.3f}.',
                     epoch + 1, epochs,
                     float(error),
                     vectors.euclideanDistance(biochemistryVector, biologyVector),
                     vectors.euclideanDistance(galaxyVector, starVector),
                     vectors.euclideanDistance(biologyVector, galaxyVector))

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
