import numpy as np
import time
import os

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import parameters
import log
import vectors
import validation
import kit


floatX = theano.config.floatX
rnd2 = lambda d0, d1: np.random.rand(d0, d1).astype(dtype=floatX)


class Model():
    def __init__(self, fileIndexMap, fileEmbeddingSize, wordIndexMap, wordEmbeddings, contexts, metricsPath,
                 epochs, batchSize, learningRate, negative, l1, l2):
        filesCount = len(fileIndexMap)
        wordsCount = len(wordIndexMap)
        wordEmbeddingSize = wordEmbeddings.shape[1]
        windowSize = contexts.shape[1] - 1
        contextSize = windowSize - 1

        fileEmbeddings = theano.shared(rnd2(filesCount, fileEmbeddingSize), 'fileEmbeddings', borrow=True)
        wordEmbeddings = theano.shared(wordEmbeddings, 'wordEmbeddings', borrow=True)
        weights = theano.shared(rnd2(fileEmbeddingSize + contextSize * wordEmbeddingSize, wordsCount), 'weights', borrow=True)

        ctxs = T.imatrix('ctxs')
        fileIndices = ctxs[:,0:1]
        wordIndices = ctxs[:,1:-1]
        targetIndices = T.reshape(ctxs[:,-1:], (ctxs.shape[0],1))

        random = RandomStreams(0)
        negativeSampleIndices = random.random_integers((targetIndices.shape[0], negative), low=0, high=wordsCount-1)
        indices = T.concatenate([targetIndices, negativeSampleIndices], axis=1)

        files = fileEmbeddings[fileIndices]
        fileFeatures = T.flatten(files, outdim=2)
        words = wordEmbeddings[wordIndices]
        wordFeatures = T.flatten(words, outdim=2)
        features = T.concatenate([fileFeatures, wordFeatures], axis=1)

        subWeights = weights[:,indices].dimshuffle(1,0,2)

        probabilities = T.batched_dot(features, subWeights)

        parameters = [fileEmbeddings, weights]

        l1Coefficient = T.scalar('l1Coefficient', dtype=floatX)
        l2Coefficient = T.scalar('l2Coefficient', dtype=floatX)

        l1Regularization = l1Coefficient * sum([abs(p).sum() for p in parameters])
        l2Regularization = l2Coefficient * sum([(p ** 2).sum() for p in parameters])

        cost = -T.mean(T.log(T.exp(probabilities[:,0])) + T.sum(T.log(T.exp(-probabilities[:,1:])), axis=1)) + \
               l1Regularization + l2Regularization

        lr = T.scalar('learningRate', dtype=floatX)

        parameters = [fileEmbeddings, weights]
        subParameters = [files]
        gradients = [T.grad(cost, wrt=subP) for subP in subParameters]
        updates = [(p, T.inc_subtensor(subP, -lr * g)) for p, subP, g in zip(parameters, subParameters, gradients)]

        parameters = [weights]
        gradients = [T.grad(cost, wrt=p) for p in parameters]
        updates = updates + [(p, p - lr * g) for p, g in zip(parameters, gradients)]

        batchIndex = T.iscalar('batchIndex')
        bs = T.iscalar('batchSize')
        trainingContexts = theano.shared(contexts, 'trainingContexts', borrow=True)

        self.trainModel = theano.function(
            inputs=[batchIndex, bs, lr, l1Coefficient, l2Coefficient],
            outputs=cost,
            updates=updates,
            givens={
                ctxs: trainingContexts[batchIndex * bs : (batchIndex + 1) * bs]
            }
        )


def train(fileIndexMap, fileEmbeddingSize, wordIndexMap, wordEmbeddings, contexts, metricsPath,
          epochs, batchSize, learningRate, negative, l1, l2):
    filesCount = len(fileIndexMap)
    wordsCount = len(wordIndexMap)
    wordEmbeddingSize = wordEmbeddings.shape[1]
    windowSize = contexts.shape[1] - 1
    contextSize = windowSize - 1

    fileEmbeddings = theano.shared(rnd2(filesCount, fileEmbeddingSize), 'fileEmbeddings', borrow=True)
    wordEmbeddings = theano.shared(wordEmbeddings, 'wordEmbeddings', borrow=True)
    weights = theano.shared(rnd2(fileEmbeddingSize + contextSize * wordEmbeddingSize, wordsCount), 'weights', borrow=True)

    ctxs = T.imatrix('ctxs')
    fileIndices = ctxs[:,0:1]
    wordIndices = ctxs[:,1:-1]
    targetIndices = T.reshape(ctxs[:,-1:], (ctxs.shape[0],1))

    random = RandomStreams(0)
    negativeSampleIndices = random.random_integers((targetIndices.shape[0], negative), low=0, high=wordsCount-1)
    indices = T.concatenate([targetIndices, negativeSampleIndices], axis=1)

    files = fileEmbeddings[fileIndices]
    fileFeatures = T.flatten(files, outdim=2)
    words = wordEmbeddings[wordIndices]
    wordFeatures = T.flatten(words, outdim=2)
    features = T.concatenate([fileFeatures, wordFeatures], axis=1)

    subWeights = weights[:,indices].dimshuffle(1,0,2)

    probabilities = T.batched_dot(features, subWeights)

    parameters = [fileEmbeddings, weights]

    l1Coefficient = T.scalar('l1Coefficient', dtype=floatX)
    l2Coefficient = T.scalar('l2Coefficient', dtype=floatX)

    l1Regularization = l1Coefficient * sum([abs(p).sum() for p in parameters])
    l2Regularization = l2Coefficient * sum([(p ** 2).sum() for p in parameters])

    cost = -T.mean(T.log(T.exp(probabilities[:,0])) + T.sum(T.log(T.exp(-probabilities[:,1:])), axis=1)) + \
           l1Regularization + l2Regularization

    lr = T.scalar('learningRate', dtype=floatX)

    parameters = [fileEmbeddings, weights]
    subParameters = [files]
    gradients = [T.grad(cost, wrt=subP) for subP in subParameters]
    updates = [(p, T.inc_subtensor(subP, -lr * g)) for p, subP, g in zip(parameters, subParameters, gradients)]

    parameters = [weights]
    gradients = [T.grad(cost, wrt=p) for p in parameters]
    updates = updates + [(p, p - lr * g) for p, g in zip(parameters, gradients)]

    batchIndex = T.iscalar('batchIndex')
    bs = T.iscalar('batchSize')
    trainingContexts = theano.shared(contexts, 'trainingContexts', borrow=True)

    trainModel = theano.function(
        inputs=[batchIndex, bs, lr, l1Coefficient, l2Coefficient],
        outputs=cost,
        updates=updates,
        givens={
            ctxs: trainingContexts[batchIndex * bs : (batchIndex + 1) * bs]
        }
    )

    startTime = time.time()
    for epoch in xrange(0, epochs):
        contextsCount = contexts.shape[0]
        batchesCount = contextsCount / batchSize + int(contextsCount % batchSize > 0)

        for bi in xrange(0, batchesCount):
            error = trainModel(bi, batchSize, learningRate, l1, l2)
            if error <= 0:
                break

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

        if error <= 0:
            break

    validation.compareEmbeddings(fileIndexMap, fileEmbeddings.get_value())
    # validation.plotEmbeddings(fileIndexMap, fileEmbeddings.get_value())


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

    log.info('Contexts loading complete. {0} contexts loaded {1} words each.', len(contexts), contextProvider.contextSize)

    train(fileIndexMap, 200, wordIndexMap, wordEmbeddings, contexts, metricsPath,
          epochs=5,
          batchSize=50,
          learningRate=0.01,
          negative=10,
          l1=0.02,
          l2=0.005)


if __name__ == '__main__':
    main()
