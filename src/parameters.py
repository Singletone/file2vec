import os
import io
import numpy

import log
import binary


class IndexContextProvider:
    def __init__(self, contextsFilePath):
        self.contextsFilePath = contextsFilePath
        self.windowsCount, self.windowSize, self.negative = self.getContextsShape()


    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop <= self.windowsCount else self.windowsCount
            step = item.step or 1

            return self.getContexts(start, stop, step)

        return self.getContexts(item, item + 1, 1)[0]


    def getContextsShape(self):
        with open(self.contextsFilePath) as contextsFile:
            contextsCount = binary.readi(contextsFile)
            contextSize = binary.readi(contextsFile)
            negative = binary.readi(contextsFile)

            return contextsCount, contextSize, negative


    def getContexts(self, start, stop, step):
        if step == 1:
            with open(self.contextsFilePath) as contextsFile:
                count = stop - start
                contextSize = self.windowSize + self.negative
                contextsSize = count * contextSize
                contextBufferSize = contextSize * 4

                # 12 for sizeof(contextsCount) + sizeof(contextSize) + sizeof(negative)
                startPosition = start * contextBufferSize + 12

                contextsFile.seek(startPosition, io.SEEK_SET)
                contexts = binary.readi(contextsFile, contextsSize)

                contexts = numpy.reshape(contexts, (count, (self.windowSize + self.negative)))
        else:
            contexts = []
            for contextIndex in xrange(start, stop, step):
                context = self[contextIndex][0]
                contexts.append(context)

        contexts = numpy.asarray(contexts, dtype='int32')

        return contexts


def dumpIndexMap(indexMap, indexMapFilePath):
    if os.path.exists(indexMapFilePath):
        os.remove(indexMapFilePath)

    with open(indexMapFilePath, 'w') as indexMapFile:
        indexMapSize = len(indexMap)
        itemIndex = 0

        binary.writei(indexMapFile, indexMapSize)

        for key, index in indexMap.items():
            keyLength = len(key)

            binary.writei(indexMapFile, keyLength)
            binary.writes(indexMapFile, key)
            binary.writei(indexMapFile, index)

            itemIndex += 1
            log.progress('Dumping index map: {0:.3f}%.', itemIndex, indexMapSize)

        indexMapFile.flush()

        log.lineBreak()


def loadIndexMap(indexMapFilePath, inverse=False):
    vocabulary = {}

    with open(indexMapFilePath, 'rb') as indexMapFile:
        itemsCount = binary.readi(indexMapFile)

        for itemIndex in range(0, itemsCount):
            wordLength = binary.readi(indexMapFile)
            word = binary.reads(indexMapFile, wordLength)
            index = binary.readi(indexMapFile)

            if inverse:
                vocabulary[index] = word
            else:
                vocabulary[word] = index

            log.progress('Loading index map: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.info('Loading index map complete. {0} items loaded.', itemsCount)

    return vocabulary


def dumpEmbeddings(embeddings, embeddingsFilePath):
    if os.path.exists(embeddingsFilePath):
        os.remove(embeddingsFilePath)

    embeddings = numpy.asarray(embeddings)
    embeddingsCount, embeddingSize = embeddings.shape

    with open(embeddingsFilePath, 'w') as embeddingsFile:
        binary.writei(embeddingsFile, embeddingsCount)
        binary.writei(embeddingsFile, embeddingSize)

        for embeddingIndex in range(0, embeddingsCount):
            embedding = embeddings[embeddingIndex]

            binary.writef(embeddingsFile, embedding)

            log.progress('Dumping embeddings: {0:.3f}%.', embeddingIndex + 1, embeddingsCount)

        log.lineBreak()


def loadEmbeddings(embeddingsFilePath):
    with open(embeddingsFilePath, 'rb') as embeddingsFile:
        embeddingsCount = binary.readi(embeddingsFile)
        embeddingSize = binary.readi(embeddingsFile)

        embeddings = numpy.empty((embeddingsCount, embeddingSize))

        for embeddingIndex in range(0, embeddingsCount):
            embedding = binary.readf(embeddingsFile, embeddingSize)
            embeddings[embeddingIndex] = embedding

            log.progress('Loading embeddings: {0:.3f}%.', embeddingIndex + 1, embeddingsCount)

        log.info('Loading embeddings complete. {0} embeddings loaded {1} features each.', embeddingsCount, embeddingSize)

        return embeddings


def loadW2VParameters(filePath, loadEmbeddings=True):
    with open(filePath, 'rb') as w2vFile:
        firstLine = w2vFile.readline()
        embeddingsCount, embeddingSize = tuple(firstLine.split(' '))
        embeddingsCount, embeddingSize = int(embeddingsCount), int(embeddingSize)
        wordIndexMap = {}
        embeddings = numpy.zeros((embeddingsCount, embeddingSize))

        embeddingIndex = 0
        while True:
            word = ''
            while True:
                char = w2vFile.read(1)

                if not char:
                    log.lineBreak()

                    if loadEmbeddings:
                        return wordIndexMap, embeddings
                    else:
                        return wordIndexMap

                if char == ' ':
                    word = word.strip()
                    break

                word += char

            wordIndexMap[word] = len(wordIndexMap)
            if loadEmbeddings:
                embedding = binary.readf(w2vFile, embeddingSize)
                embeddings[wordIndexMap[word]] = embedding
            else:
                w2vFile.seek(embeddingSize * 4, io.SEEK_CUR)

            embeddingIndex += 1
            log.progress('Loading W2V embeddings: {0:.3f}%. {1} embeddings {2} features each.',
                         embeddingIndex,
                         embeddingsCount,
                         embeddingsCount,
                         embeddingSize)