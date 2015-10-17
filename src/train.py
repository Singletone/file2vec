from theano.tensor.shared_randomstreams import RandomStreams

import numpy
import theano
import theano.tensor as T

import log
import parameters
import validation


class ParagraphVectorsModel:
    def __init__(self, wordEmbeddings, fileVocabularySize, fileEmbeddingSize, negativeSamplesCount, contextSize):
        floatX = theano.config.floatX
        empty = lambda *shape: numpy.empty(shape, dtype='int32')

        wordEmbeddingsCount, wordEmbeddingSize = wordEmbeddings.shape

        defaultFileEmbeddings = numpy.random.randn(fileVocabularySize, fileEmbeddingSize).astype(dtype=floatX)
        self.fileEmbeddings = theano.shared(defaultFileEmbeddings, name='fileEmbeddings', borrow=True)

        defaultWordEmbeddings = numpy.asarray(wordEmbeddings, dtype=floatX)
        self.wordEmbeddings = theano.shared(defaultWordEmbeddings, name='wordEmbeddings', borrow=True)

        defaultWeight = numpy.random.randn(fileEmbeddingSize + contextSize * wordEmbeddingSize, wordEmbeddingsCount).astype(dtype=floatX)
        self.weight = theano.shared(defaultWeight, name='weight', borrow=True)

        defaultBias = numpy.random.randn(negativeSamplesCount + 1).astype(dtype=floatX)
        self.bias = theano.shared(defaultBias, name='bias', borrow=True)

        parameters = [self.fileEmbeddings, self.weight, self.bias]

        fileIndex = T.ivector('fileIndex')
        contextIndices = T.imatrix('contextIndices')
        targetWordIndex = T.ivector('targetWordIndex')

        # assert output is a single file embedding
        fileEmbedding = self.fileEmbeddings[indexContext]

        # assert output is N word embeddings
        contextEmbeddings = self.wordEmbeddings[indexContext]

        # assert output is same word embeddings in the same order
        contextEmbeddings = T.flatten(contextEmbeddings)

        # assert concatenation gives file embedding followed by flatten word embeddings
        embeddings = T.concatenate([fileEmbedding, contextEmbeddings])

        random = RandomStreams()
        negativeSampleIndices = random.random_integers(negativeSamplesCount, 0, wordEmbeddingsCount - 1, dtype='int32')

        targetWordIndices = T.concatenate([targetWordIndex, negativeSampleIndices])
        # assert indexing will produce sub weight matrix of needed shape
        subWeight = self.weight[:,targetWordIndices]

        probabilities = T.nnet.softmax(T.dot(context, self.weight) + self.bias)
        targetProbability = T.ivector('targetProbability')

        cost = -T.mean(T.log(probabilities)[T.arange(targetProbability.shape[0]), 0])

        learningRate = T.scalar('learningRate', dtype=floatX)

        gradients = [T.grad(cost, wrt=p) for p in parameters]
        updates = [(p, p - learningRate * g) for p, g in zip(parameters, gradients)]

        miniBatchIndex = T.lscalar('miniBatchIndex')
        miniBatchSize = T.iscalar('miniBatchSize')

        self.fileInput = theano.shared(empty(1), borrow=True)
        self.contextInput = theano.shared(empty(1,1), borrow=True)
        self.targetOutput = theano.shared(empty(1), borrow=True)

        self.trainModel = theano.function(
            inputs=[miniBatchIndex, miniBatchSize, learningRate],
            outputs=cost,
            updates=updates,
            givens={
                fileIndex: self.fileInput[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize],
                contextIndices: self.contextInput[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize],
                targetProbability: self.targetOutput[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
            }
        )

        self.getFileEmbeddings = theano.function(
            inputs=[fileEmbeddingIndices],
            outputs=fileEmbeddingSample
        )


    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else self.fileVocabularySize
            step = item.step if item.step is not None else 1

            if abs(stop) > self.fileVocabularySize:
                stop = self.fileVocabularySize

            indices = [i for i in xrange(start, stop, step)]

            return self.getFileEmbeddings(indices)

        return self.getFileEmbeddings([item])[0]


    def train(self, fileIndices, wordIndices, trainingTargetOutput, miniBatchSize, learningRate):
        asarray = lambda x: numpy.asarray(x, dtype='int32')

        wordIndices = asarray(wordIndices)
        trainingTargetOutput = asarray(trainingTargetOutput)

        self.fileInput.set_value(fileIndices)
        self.contextInput.set_value(wordIndices)
        self.targetOutput.set_value(trainingTargetOutput)

        trainInputSize = wordIndices.shape[0]
        trainingBatchesCount = trainInputSize / miniBatchSize + int(trainInputSize % miniBatchSize > 0)

        for trainingBatchIndex in xrange(0, trainingBatchesCount):
            self.trainModel(trainingBatchIndex, miniBatchSize, learningRate)



    def dump(self, fileEmbeddingsPath):
        fileEmbeddings = self.fileEmbeddings.get_value()
        parameters.dumpEmbeddings(fileEmbeddings, fileEmbeddingsPath)


def trainFileEmbeddings(fileIndexMapFilePath, wordIndexMapFilePath, wordEmbeddingsFilePath, contextsPath,
                        fileEmbeddingsPath, fileEmbeddingSize, epochs, superBatchSize, miniBatchSize, learningRate, negativeSamplesCount):
    fileVocabulary = parameters.loadIndexMap(fileIndexMapFilePath)
    fileVocabularySize = len(fileVocabulary)

    wordVocabulary = parameters.loadIndexMap(wordIndexMapFilePath)
    wordVocabularySize = len(wordVocabulary)

    wordEmbeddings = parameters.loadEmbeddings(wordEmbeddingsFilePath)

    contextProvider = parameters.IndexContextProvider(contextsPath)
    contextsCount, contextSize = contextProvider.contextsCount, contextProvider.contextSize

    model = ParagraphVectorsModel(wordEmbeddings, fileVocabularySize, fileEmbeddingSize, negativeSamplesCount, contextSize - 1)
    fileEmbeddingsBefore = model[:]

    superBatchesCount = contextsCount / superBatchSize + 1

    for epoch in xrange(0, epochs):
        for superBatchIndex in xrange(0, superBatchesCount):
            contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

            fileIndices, wordIndices, targetWordIndicis = contextSuperBatch[:,0], contextSuperBatch[:,1:-1], contextSuperBatch[:,:-1]

            model.train(fileIndices, wordIndices, targetWordIndicis, miniBatchSize, learningRate)

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
        wordIndexMapFilePath= '../data/Drosophila/Parameters/word_index_map.bin',
        wordEmbeddingsFilePath = '../data/Drosophila/Parameters/word_embeddings.bin',
        contextsPath = '../data/Drosophila/Processed/contexts.bin',
        fileEmbeddingsPath = '../data/Drosophila/Parameters/file_embeddings.bin',
        fileEmbeddingSize = 100,
        epochs = 100,
        superBatchSize = 1000,
        miniBatchSize = 50,
        learningRate = 0.13,
        negativeSamplesCount = 5)

