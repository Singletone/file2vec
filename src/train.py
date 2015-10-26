import numpy as np
import time
import os

import theano
import theano.tensor as T

import parameters
import log
import vectors
import validation
import kit

floatX = theano.config.floatX
empty = lambda *shape: np.empty(shape, dtype='int32')
rnd1 = lambda d0: np.random.rand(d0).astype(dtype=floatX)
rnd2 = lambda d0, d1: np.random.rand(d0, d1).astype(dtype=floatX)


def train(fileIndexMap, wordIndexMap, wordEmbeddings, contexts, metricsPath):
    contextsCount = len(contexts)

    filesCount = len(fileIndexMap)
    fileEmbeddingSize = 400
    wordsCount = len(wordIndexMap)
    wordEmbeddingSize = wordEmbeddings.shape[1]
    windowSize = contexts.shape[1] - 1
    contextSize = windowSize - 1

    fileEmbeddings = theano.shared(rnd2(filesCount, fileEmbeddingSize), 'fileEmbeddings', borrow=True)
    wordEmbeddings = theano.shared(wordEmbeddings, 'wordEmbeddings', borrow=True)
    weights = theano.shared(rnd2(fileEmbeddingSize + contextSize * wordEmbeddingSize, wordsCount), 'weights', borrow=True)
    biases = theano.shared(rnd1(wordsCount), 'bias', borrow=True)

    ctxs = T.imatrix('ctxs')
    fileIndices = ctxs[:,0:1]
    wordIndices = ctxs[:,1:-1]
    targetIndices = T.reshape(ctxs[:,-1:], (ctxs.shape[0],))

    files = fileEmbeddings[fileIndices]
    fileFeatures = T.flatten(files, outdim=2)
    words = wordEmbeddings[wordIndices]
    wordFeatures = T.flatten(words, outdim=2)
    features = T.concatenate([fileFeatures, wordFeatures], axis=1)

    probabilities = T.nnet.softmax(T.dot(features, weights) + biases)

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

    gradient = T.grad(cost=cost, wrt=files)
    updates = [(fileEmbeddings, T.inc_subtensor(files, -learningRate * gradient))] + updates

    batchIndex = T.iscalar('batchIndex')
    batchSize = T.iscalar('batchSize')
    trainingContexts = theano.shared(contexts, 'trainingContexts', borrow=True)

    trainModel = theano.function(
        inputs=[batchIndex, batchSize, learningRate, l1Coefficient, l2Coefficient],
        outputs=cost,
        updates=updates,
        givens={
            ctxs: trainingContexts[batchIndex * batchSize : (batchIndex + 1) * batchSize]
        }
    )

    startTime = time.time()
    epochs = 50
    bs = 100 # bs for batchSize
    for epoch in xrange(0, epochs):
        contextsCount = contexts.shape[0]
        batchesCount = contextsCount / bs + int(contextsCount % bs > 0)

        for bi in xrange(0, batchesCount): # bi for batchIndex
            error = trainModel(bi, bs, 0.4, 0.006, 0.001)

        fe = fileEmbeddings.get_value()
        we = lambda key: fe[fileIndexMap['../data/Duplicates/Prepared/duplicates/{0}.txt'.format(key)]]
        distance = lambda left, right: vectors.euclideanDistance(we(left), we(right))

        metrics = {
            'tanks': distance('tank_0', 'tank_1'),
            'viruses': distance('virus_0', 'virus_1'),
            'tankVirus': distance('tank_0', 'virus_0'),
            'error': error
        }

        validation.dump(metricsPath, epoch, metrics)

        elapsed = time.time() - startTime
        secondsPerEpoch = elapsed / (epoch + 1)

        log.progress('Training model: {0:.3f}%. {1:.3f} sec per epoch. Error: {2:.3f}. Tanks={3:.3f}. Viruses={4:.3f}. Tank/Virus={5:.3f}.',
                     epoch + 1, epochs,
                     secondsPerEpoch,
                     float(error),
                     metrics['tanks'],
                     metrics['viruses'],
                     metrics['tankVirus'])

    validation.compareEmbeddings(fileIndexMap, fileEmbeddings.get_value())


def main():
    pathTo = kit.PathTo('Duplicates')

    fileIndexMap = parameters.loadIndexMap(pathTo.fileIndexMap)
    wordIndexMap = parameters.loadIndexMap(pathTo.wordIndexMap)
    indexWordMap = parameters.loadIndexMap(pathTo.wordIndexMap, inverse=True)
    wordEmbeddings = parameters.loadEmbeddings(pathTo.wordEmbeddings)
    metricsPath = pathTo.metrics('history.csv')

    if os.path.exists(metricsPath):
        os.remove(metricsPath)

    contextProvider = parameters.IndexContextProvider(pathTo.contexts)
    contexts = contextProvider[:]

    log.info('Contexts loading complete. {0} contexts loaded {1} words each.', len(contexts), contextProvider.windowSize)

    train(fileIndexMap, wordIndexMap, wordEmbeddings, contexts, metricsPath)


if __name__ == '__main__':
    main()
