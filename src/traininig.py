import numpy as np
import time
import os

import theano
import theano.tensor as T

import parameters
import log
import validation
import kit
import binary


floatX = theano.config.floatX
empty = lambda *shape: np.empty(shape, dtype='int32')
rnd2 = lambda d0, d1: np.random.rand(d0, d1).astype(dtype=floatX)


class Model:
    def __init__(self, fileEmbeddings, wordEmbeddings, weights=None, contextSize=None, negative=None):
        filesCount, fileEmbeddingSize = fileEmbeddings.shape
        wordsCount, wordEmbeddingSize = wordEmbeddings.shape

        if weights is not None:
            featuresCount, activationsCount = weights.shape
            contextSize = (featuresCount - fileEmbeddingSize) / wordEmbeddingSize
            negative = activationsCount - 1
        else:
            weights = rnd2(fileEmbeddingSize + contextSize * wordEmbeddingSize, wordsCount)

        self.fileEmbeddings = theano.shared(fileEmbeddings, 'fileEmbeddings', borrow=True)
        self.wordEmbeddings = theano.shared(wordEmbeddings, 'wordEmbeddings', borrow=True)
        self.weights = theano.shared(weights, 'weights', borrow=True)

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
        subParameters = [files, None]

        l1Coef = T.scalar('l1Coefficient', dtype=floatX)
        l2Coef = T.scalar('l2Coefficient', dtype=floatX)

        l1 = l1Coef * abs(self.weights).sum()
        l2 = l2Coef * (self.weights ** 2).sum()

        cost = -T.mean(T.log(T.nnet.sigmoid(probabilities[:,0])) + T.sum(T.log(T.nnet.sigmoid(-probabilities[:,1:])), axis=1)) \
               + l1 + l2

        learningRate = T.scalar('learningRate', dtype=floatX)

        updates = []
        for p, subP in zip(parameters, subParameters):
            if subP is not None:
                gradient = T.grad(cost, wrt=subP)
                update = (p, T.inc_subtensor(subP, -learningRate * gradient))
            else:
                gradient = T.grad(cost, wrt=p)
                update = (p, p - learningRate * gradient)

            updates.append(update)

        batchIndex = T.iscalar('batchIndex')
        batchSize = T.iscalar('batchSize')
        self.trainingContexts = theano.shared(empty(1,1), 'trainingContexts', borrow=True)

        self.trainModel = theano.function(
            inputs=[batchIndex, batchSize, learningRate, l1Coef, l2Coef],
            outputs=cost,
            updates=updates,
            givens={
                contexts: self.trainingContexts[batchIndex * batchSize : (batchIndex + 1) * batchSize]
            }
        )


    def dump(self, fileEmbeddingsPath, weightsPath):
        fileEmbeddings = self.fileEmbeddings.get_value()
        binary.dumpMatrix(fileEmbeddingsPath, fileEmbeddings)

        weights = self.weights.get_value()
        binary.dumpMatrix(weightsPath, weights)


    @staticmethod
    def load(fileEmbeddingsPath, wordEmbeddingsPath, weightsPath):
        fileEmbeddings = binary.loadMatrix(fileEmbeddingsPath)
        wordEmbeddings = binary.loadMatrix(wordEmbeddingsPath)
        weights = binary.loadMatrix(weightsPath)

        filesCount, fileEmbeddingSize = fileEmbeddings.shape
        wordsCount, wordEmbeddingSize = wordEmbeddings.shape
        featuresCount, activationsCount = weights.shape
        contextSize = (featuresCount - fileEmbeddingSize) / wordEmbeddingSize
        negative = activationsCount - 1

        return Model(filesCount, fileEmbeddingSize, wordEmbeddings, contextSize, negative)



def train(model, fileIndexMap, wordIndexMap, wordEmbeddings, contexts, metricsPath,
          epochs, batchSize, learningRate, l1, l2):
    model.trainingContexts.set_value(contexts)

    contextsCount, contextSize = contexts.shape

    batchesCount = contextsCount / batchSize + int(contextsCount % batchSize > 0)

    startTime = time.time()
    errors = []
    for epoch in xrange(0, epochs):
        error = 0
        for batchIndex in xrange(0, batchesCount):
            error += model.trainModel(batchIndex, batchSize, learningRate, l1, l2)

        error = error / batchesCount
        errors.append(error)

        elapsed = time.time() - startTime

        log.progress('Training model: {0:.3f}%. Epoch: {1}. Elapsed: {2}. Error: {3:.3f}. Learning rate: {4}.',
                     epoch + 1,
                     epochs,
                     epoch + 1,
                     log.delta(elapsed),
                     error,
                     learningRate)

        metrics = {
            'error': error,
            'learningRate': learningRate
        }

        validation.dump(metricsPath, epoch, metrics)

    validation.compareEmbeddings(fileIndexMap, model.fileEmbeddings.get_value(), annotate=True)
    validation.plotEmbeddings(fileIndexMap, model.fileEmbeddings.get_value())
    validation.compareMetrics(metricsPath, 'error')


def launch(pathTo):
    fileIndexMap = parameters.loadIndexMap(pathTo.fileIndexMap)
    filesCount = len(fileIndexMap)
    fileEmbeddingSize = 800
    wordIndexMap = parameters.loadIndexMap(pathTo.wordIndexMap)
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

    fileEmbeddings = rnd2(filesCount, fileEmbeddingSize)
    model = Model(fileEmbeddings, wordEmbeddings, contextSize=contextSize, negative=negative)

    train(model, fileIndexMap, wordIndexMap, wordEmbeddings, contexts, metricsPath,
          epochs=30,
          batchSize=1,
          learningRate=0.01,
          l1=0.02,
          l2=0.005)

    model.dump(pathTo.fileEmbeddings, pathTo.weights)


if __name__ == '__main__':
    pathTo = kit.PathTo('Duplicates')
    launch(pathTo)