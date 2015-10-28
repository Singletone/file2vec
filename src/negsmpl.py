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
import binary as bin


floatX = theano.config.floatX
empty = lambda *shape: np.empty(shape, dtype='int32')
rnd2 = lambda d0, d1: np.random.rand(d0, d1).astype(dtype=floatX)


class Model():
    def __init__(self, filesCount, fileEmbeddingSize, wordEmbeddings, contextSize, negative):
        wordsCount, wordEmbeddingSize = wordEmbeddings.shape

        self.fileEmbeddings = theano.shared(rnd2(filesCount, fileEmbeddingSize), 'fileEmbeddings', borrow=True)
        self.wordEmbeddings = theano.shared(wordEmbeddings, 'wordEmbeddings', borrow=True)
        self.weights = theano.shared(rnd2(fileEmbeddingSize + contextSize * wordEmbeddingSize, wordsCount), 'weights', borrow=True)

        fileIndicesOffset = 0
        wordIndicesOffset = fileIndicesOffset + 1
        indicesOffset = wordIndicesOffset + contextSize

        contexts = T.imatrix('contexts')
        fileIndices = contexts[:,fileIndicesOffset:wordIndicesOffset]
        wordIndices = contexts[:,wordIndicesOffset:indicesOffset]
        indices = contexts[:,indicesOffset:indicesOffset + negative]

        files = self.fileEmbeddings[fileIndices]
        fileFeatures = T.flatten(files, outdim=2)
        words = self.wordEmbeddings[wordIndices]
        wordFeatures = T.flatten(words, outdim=2)
        features = T.concatenate([fileFeatures, wordFeatures], axis=1)

        subWeights = self.weights[:,indices].dimshuffle(1, 0, 2)

        probabilities = T.batched_dot(features, subWeights)

        parameters = [self.fileEmbeddings, self.weights]

        l1Coefficient = T.scalar('l1Coefficient', dtype=floatX)
        l2Coefficient = T.scalar('l2Coefficient', dtype=floatX)

        l1Regularization = l1Coefficient * sum([abs(p).sum() for p in parameters])
        l2Regularization = l2Coefficient * sum([(p ** 2).sum() for p in parameters])

        cost = -T.mean(T.log(T.exp(probabilities[:,0])) + T.sum(T.log(T.exp(-probabilities[:,1:])), axis=1)) + \
               l1Regularization + l2Regularization

        lr = T.scalar('learningRate', dtype=floatX)

        parameters = [self.fileEmbeddings]
        subParameters = [files]
        gradients = [T.grad(cost, wrt=subP) for subP in subParameters]
        updates = [(p, T.inc_subtensor(subP, -lr * g)) for p, subP, g in zip(parameters, subParameters, gradients)]

        parameters = [self.weights]
        gradients = [T.grad(cost, wrt=p) for p in parameters]
        updates = updates + [(p, p - lr * g) for p, g in zip(parameters, gradients)]

        batchIndex = T.iscalar('batchIndex')
        bs = T.iscalar('batchSize')
        self.trainingContexts = theano.shared(empty(1,1), 'trainingContexts', borrow=True)

        self.trainModel = theano.function(
            inputs=[batchIndex, bs, lr, l1Coefficient, l2Coefficient],
            outputs=cost,
            updates=updates,
            givens={
                contexts: self.trainingContexts[batchIndex * bs : (batchIndex + 1) * bs]
            }
        )


    def dump(self, fileEmbeddingsPath, weightsPath):
        fileEmbeddings = self.fileEmbeddings.get_value()
        bin.dumpMatrix(fileEmbeddingsPath, fileEmbeddings)

        weights = self.weights.get_value()
        bin.dumpMatrix(weightsPath, weights)


    @staticmethod
    def load(fileEmbeddingsPath, wordEmbeddingsPath, weightsPath):
        fileEmbeddings = bin.loadMatrix(fileEmbeddingsPath)
        wordEmbeddings = bin.loadMatrix(wordEmbeddingsPath)
        weights = bin.loadMatrix(weightsPath)

        filesCount, fileEmbeddingSize = fileEmbeddings.shape
        wordsCount, wordEmbeddingSize = wordEmbeddings.shape
        featuresCount, activationsCount = weights.shape
        contextSize = (featuresCount - fileEmbeddingSize) / wordEmbeddingSize
        negative = activationsCount - 1

        return Model(filesCount, fileEmbeddingSize, wordEmbeddings, contextSize, negative)



def train(model, fileIndexMap, fileEmbeddingSize, wordIndexMap, wordEmbeddings, contexts, metricsPath,
          epochs, batchSize, learningRate, negative, l1, l2):
    model.trainingContexts.set_value(contexts)

    startTime = time.time()
    contextsCount = contexts.shape[0]
    batchesCount = contextsCount / batchSize + int(contextsCount % batchSize > 0)

    for epoch in xrange(0, epochs):
        for batchIndex in xrange(0, batchesCount):
            error = model.trainModel(batchIndex, batchSize, learningRate, l1, l2)
            if error <= 0:
                break

        fe = model.fileEmbeddings.get_value()
        we = lambda key: fe[fileIndexMap['../data/Duplicates/Prepared/duplicates/{0}.txt'.format(key)]]
        distance = lambda left, right: vectors.euclideanDistance(we(left), we(right))

        metrics = {
            'tanks': distance('tank_0', 'tank_1'),
            'alexes': distance('alex_0', 'alex_1'),
            'tankAlex': distance('tank_0', 'alex_0'),
            'carAlex': distance('car_0', 'alex_0'),
            'spiderNasa': distance('spider_0', 'nasa_0'),
            'error': error
        }

        validation.dump(metricsPath, epoch, metrics)

        elapsed = time.time() - startTime
        secondsPerEpoch = elapsed / (epoch + 1)

        log.progress('Training model: {0:.3f}%. {1:.3f} sec per epoch. Error: {2:.3f}. Spiders={3:.3f}. Alexes={4:.3f}. Tank/Alex={5:.3f}. Car/Alex={6:.3f}. Spider/Nasa={7:.3f}.',
                     epoch + 1, epochs,
                     secondsPerEpoch,
                     float(error),
                     metrics['tanks'],
                     metrics['alexes'],
                     metrics['tankAlex'],
                     metrics['carAlex'],
                     metrics['spiderNasa'])

        if error <= 0:
            break

    validation.compareEmbeddings(fileIndexMap, model.fileEmbeddings.get_value())
    # validation.plotEmbeddings(fileIndexMap, model.fileEmbeddings.get_value())


def main():
    pathTo = kit.PathTo('Duplicates')

    fileIndexMap = parameters.loadIndexMap(pathTo.fileIndexMap)
    filesCount = len(fileIndexMap)
    wordIndexMap = parameters.loadIndexMap(pathTo.wordIndexMap)
    indexWordMap = parameters.loadIndexMap(pathTo.wordIndexMap, inverse=True)
    wordEmbeddings = parameters.loadEmbeddings(pathTo.wordEmbeddings)
    metricsPath = pathTo.metrics('history.csv')

    if os.path.exists(metricsPath):
        os.remove(metricsPath)

    contextProvider = parameters.IndexContextProvider(pathTo.contexts)
    windowSize = contextProvider.windowSize - 1
    contextSize = windowSize - 1
    negative = contextProvider.negative
    contexts = contextProvider[:]

    log.info('Contexts loading complete. {0} contexts loaded {1} words and {2} negative samples each.',
             len(contexts),
             contextProvider.windowSize,
             contextProvider.negative)

    model = Model(filesCount, 200, wordEmbeddings, contextSize, negative)

    train(model, fileIndexMap, 200, wordIndexMap, wordEmbeddings, contexts, metricsPath,
          epochs=2,
          batchSize=50,
          learningRate=0.01,
          negative=10,
          l1=0.02,
          l2=0.005)

    model.dump(pathTo.fileEmbeddings, pathTo.weights)


if __name__ == '__main__':
    main()
