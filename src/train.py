from numpy.random import randn as random
import numpy
import theano
import theano.tensor as T

import parameters
import validation


class ParagraphVectorsModel:
    def __init__(self, wordEmbeddings, contextSize, fileVocabularySize, fileEmbeddingSize):
        floatX = theano.config.floatX

        self.fileVocabularySize = fileVocabularySize

        defaultFileEmbeddings = random(fileVocabularySize, fileEmbeddingSize).astype(dtype=floatX)
        self.fileEmbeddings = theano.shared(defaultFileEmbeddings, name='fileEmbeddings', borrow=True)

        self.wordEmbeddings = theano.shared(wordEmbeddings, name='wordEmbeddings', borrow=True)

        fileEmbeddingIndices = T.ivector('fileEmbeddingIndices')
        fileEmbeddingSample = self.fileEmbeddings[fileEmbeddingIndices]

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


    def train(self, fileIndices, wordIndices, targetWordIndices, miniBatchSize, learningRate):
        pass


    def dump(self, fileEmbeddingsPath):
        fileEmbeddings = self.fileEmbeddings.get_value()
        parameters.dumpEmbeddings(fileEmbeddings, fileEmbeddingsPath)


def trainFileEmbeddings(fileVocabularyPath, wordVocabularyPath, wordEmbeddingsFilePath, contextsPath,
                        fileEmbeddingsPath, fileEmbeddingSize, epochs, superBatchSize, miniBatchSize, learningRate):
    fileVocabulary = parameters.loadFileVocabulary(fileVocabularyPath)
    fileVocabularySize = len(fileVocabulary)

    wordVocabulary = parameters.loadWordVocabulary(wordVocabularyPath)
    wordVocabularySize = len(wordVocabulary)

    wordEmbeddings = parameters.loadEmbeddigns(wordEmbeddingsFilePath)

    contextProvider = parameters.IndexContextProvider(contextsPath)
    contextsCount, contextSize = contextProvider.contextsCount, contextProvider.contextSize

    model = ParagraphVectorsModel(wordEmbeddings, contextSize, fileVocabularySize, fileEmbeddingSize)
    superBatchesCount = contextsCount / superBatchSize + 1

    for epoch in xrange(0, epochs):
        for superBatchIndex in xrange(0, superBatchesCount):
            contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

            fileIndices, wordIndices, targetWordIndices = contextSuperBatch[:,1], contextSuperBatch[:,1:-1], contextSuperBatch[:,-1]

            model.train(fileIndices, wordIndices, targetWordIndices, miniBatchSize, learningRate)

    model.dump(fileEmbeddingsPath)

    fileEmbeddings = model[:]
    validation.plotEmbeddings(fileVocabulary, fileEmbeddings)

    model.dump(fileEmbeddingsPath)


if __name__ == '__main__':
    fileVocabularyPath = '../data/Drosophila/Parameters/file_vocabulary.bin'
    wordVocabularyPath = '../data/Drosophila/Parameters/word_vocabulary.bin'
    wordEmbeddingsFilePath = '../data/Drosophila/Parameters/word_embeddings.bin'
    contextsPath = '../data/Drosophila/Processed/contexts.bin'
    fileEmbeddingsPath = '../data/Drosophila/Parameters/file_embeddings.bin'
    fileEmbeddingSize = 200

    trainFileEmbeddings(
        fileVocabularyPath,
        wordVocabularyPath,
        wordEmbeddingsFilePath,
        contextsPath,
        fileEmbeddingsPath,
        fileEmbeddingSize,
        epochs = 100,
        superBatchSize = 1000,
        miniBatchSize = 50,
        learningRate=0.13)

