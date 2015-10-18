import numpy
import theano
import theano.tensor as T

import log
import parameters
import validation


class ParagraphVectorModel():
    def __init__(self, wordEmbeddings, filesCount, fileEmbeddingSize, windowSize):
        floatX = theano.config.floatX
        empty = lambda *shape: numpy.empty(shape, dtype='int32')
        rnd1 = lambda d0: numpy.random.rand(d0).astype(dtype=floatX)
        rnd2 = lambda d0, d1: numpy.random.rand(d0, d1).astype(dtype=floatX)

        wordsCount, wordEmbeddingSize = wordEmbeddings.shape
        self.filesCount = filesCount

        self.fileEmbeddings = theano.shared(rnd2(filesCount, fileEmbeddingSize), 'fileEmbeddings', borrow=True)
        self.wordEmbeddings = theano.shared(wordEmbeddings, 'wordEmbeddings', borrow=True)
        self.weights = theano.shared(rnd2(fileEmbeddingSize + (windowSize - 2) * wordEmbeddingSize, wordsCount), 'weights', borrow=True)
        self.biases = theano.shared(rnd1(wordsCount), 'bias', borrow=True)

        input = T.ivector('input')
        fileIndex = input[0:1]
        wordIndices = input[1:-1]
        targetWordIndex = input[-1:]

        file = self.fileEmbeddings[fileIndex]
        words = self.wordEmbeddings[wordIndices]
        embeddings = T.concatenate([file, words], axis=0)
        contextEmbeddings = T.reshape(embeddings, (embeddings.shape[0] * embeddings.shape[1],))

        probabilities = T.nnet.softmax(T.dot(contextEmbeddings, self.weights) + self.biases)

        cost = -T.mean(T.log(probabilities)[T.arange(probabilities.shape[0]), targetWordIndex])

        parameters = [self.fileEmbeddings, self.weights, self.biases]
        gradients = [T.grad(cost, wrt=sp) for sp in parameters]
        learningRate = T.scalar('learningRate', dtype=floatX)
        updates = [(p, p - learningRate * g) for p, g in zip(parameters, gradients)]

        contextIndex = T.iscalar('contextIndex')

        self.contexts = theano.shared(empty(1,1), borrow=True)
        self.trainModel = theano.function(
            inputs=[contextIndex, learningRate],
            outputs=cost,
            updates=updates,
            givens={
                input: self.contexts[contextIndex]
            })

        i = T.ivector('i')
        sample = self.fileEmbeddings[i]

        self.getFileEmbeddings = theano.function(
            inputs=[i],
            outputs=sample
        )


    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else self.filesCount
            step = item.step if item.step is not None else 1

            if abs(stop) > self.filesCount:
                stop = self.filesCount

            indices = [i for i in xrange(start, stop, step)]

            return self.getFileEmbeddings(indices)

        return self.getFileEmbeddings([item])[0]


    def train(self, contexts, miniBatchSize, learningRate):
        self.contexts.set_value(contexts)

        for contextIndex in xrange(0, len(contexts)):
            self.trainModel(contextIndex, learningRate)


    def dump(self, embeddingsFilePath):
        fileEmbeddings = self.fileEmbeddings.get_value()
        parameters.dumpEmbeddings(fileEmbeddings, embeddingsFilePath)


def trainFileEmbeddings(fileIndexMapFilePath, wordEmbeddingsFilePath, contextsPath,
                        fileEmbeddingsPath, fileEmbeddingSize, epochs, superBatchSize, miniBatchSize, learningRate):
    fileVocabulary = parameters.loadIndexMap(fileIndexMapFilePath)
    filesCount = len(fileVocabulary)

    wordEmbeddings = parameters.loadEmbeddings(wordEmbeddingsFilePath)

    contextProvider = parameters.IndexContextProvider(contextsPath)
    contextsCount, windowSize = contextProvider.contextsCount, contextProvider.contextSize

    model = ParagraphVectorModel(wordEmbeddings, filesCount, fileEmbeddingSize, windowSize)
    fileEmbeddingsBefore = model[:]

    superBatchesCount = contextsCount / superBatchSize + 1

    for epoch in xrange(0, epochs):
        for superBatchIndex in xrange(0, superBatchesCount):
            contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

            model.train(contextSuperBatch, miniBatchSize, learningRate)

            log.progress('Training model: {0:.3f}%.',
                         epoch * superBatchesCount + superBatchIndex + 1,
                         epochs * superBatchesCount)

    log.lineBreak()

    model.dump(fileEmbeddingsPath)

    fileEmbeddingsAfter = model[:]
    validation.compareEmbeddings(fileVocabulary, [fileEmbeddingsBefore, fileEmbeddingsAfter])

    model.dump(fileEmbeddingsPath)


if __name__ == '__main__':
    trainFileEmbeddings(
        fileIndexMapFilePath= '../data/Drosophila/Parameters/file_index_map.bin',
        wordEmbeddingsFilePath = '../data/Drosophila/Parameters/word_embeddings.bin',
        contextsPath = '../data/Drosophila/Processed/contexts.bin',
        fileEmbeddingsPath = '../data/Drosophila/Parameters/file_embeddings.bin',
        fileEmbeddingSize = 100,
        epochs = 5,
        superBatchSize = 10000,
        miniBatchSize = 100,
        learningRate = 0.13)
