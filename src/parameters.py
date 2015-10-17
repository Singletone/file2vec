import os
import io
import struct
import numpy

import log


class IndexContextProvider():
    def __init__(self, contextsFilePath):
        self.contextsFilePath = contextsFilePath
        self.contextsCount, self.contextSize = self.getContextsShape()


    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop <= self.contextsCount else self.contextsCount
            step = item.step or 1

            return self.getContexts(start, stop, step)

        return self.getContexts(item, item + 1, 1)


    def getContextsShape(self):
        with open(self.contextsFilePath) as contextsFile:
            contextsCount = contextsFile.read(4)
            contextSize = contextsFile.read(4)

            contextsCount = struct.unpack('i', contextsCount)[0]
            contextSize = struct.unpack('i', contextSize)[0]

            return contextsCount, contextSize


    def getContexts(self, start, stop, step):
        if step == 1:
            with open(self.contextsFilePath) as contextsFile:
                count = stop - start
                contextBufferSize = self.contextSize * 4
                contextsBufferSize = count * contextBufferSize
                startPosition = start * contextBufferSize + 8 # 8 for contextsCount + contextSize

                contextsFile.seek(startPosition, io.SEEK_SET)
                contextsBuffer = contextsFile.read(contextsBufferSize)

                contextFormat = '{0}i'.format(self.contextSize * count)
                contexts = struct.unpack(contextFormat, contextsBuffer)

                contexts = numpy.reshape(contexts, (count, self.contextSize))
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

        indexMapFile.write(struct.pack('i', indexMapSize))

        for key, index in indexMap.items():
            keyLength = struct.pack('i', len(key))
            index = struct.pack('i', index)

            indexMapFile.write(keyLength)
            indexMapFile.write(key)
            indexMapFile.write(index)

            itemIndex += 1
            log.progress('Dumping index map: {0:.3f}%.', itemIndex, indexMapSize)

        indexMapFile.flush()

        log.lineBreak()


def loadIndexMap(indexMapFilePath, inverse=False):
    vocabulary = {}

    with open(indexMapFilePath, 'rb') as indexMapFile:
        itemsCount = indexMapFile.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        for itemIndex in range(0, itemsCount):
            wordLength = indexMapFile.read(4)
            wordLength = struct.unpack('i', wordLength)[0]

            word = indexMapFile.read(wordLength)

            index = indexMapFile.read(4)
            index = struct.unpack('i', index)[0]

            if inverse:
                vocabulary[index] = word
            else:
                vocabulary[word] = index

            log.progress('Loading index map: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.lineBreak()

    return vocabulary


def dumpEmbeddings(embeddings, embeddingsFilePath):
    if os.path.exists(embeddingsFilePath):
        os.remove(embeddingsFilePath)

    embeddingsCount, embeddingSize = embeddings.shape

    with open(embeddingsFilePath, 'w') as embeddingsFile:
        embeddingsFile.write(struct.pack('i', embeddingsCount))
        embeddingsFile.write(struct.pack('i', embeddingSize))

        format = '{0}f'.format(embeddingSize)
        for embeddingIndex in range(0, embeddingsCount):
            embedding = embeddings[embeddingIndex]
            embedding = struct.pack(format, *embedding)

            embeddingsFile.write(embedding)

            log.progress('Dumping embeddings: {0:.3f}%.', embeddingIndex + 1, embeddingsCount)


def loadEmbeddings(embeddingsFilePath):
    with open(embeddingsFilePath, 'rb') as embeddingsFile:
        embeddingsCount = embeddingsFile.read(4)
        embeddingsCount = struct.unpack('i', embeddingsCount)[0]

        embeddingSize = embeddingsFile.read(4)
        embeddingSize = struct.unpack('i', embeddingSize)[0]

        embeddings = numpy.empty((embeddingsCount, embeddingSize))

        format = '{0}f'.format(embeddingSize)
        for embeddingIndex in range(0, embeddingsCount):
            embedding = embeddingsFile.read(embeddingSize * 4)
            embedding = struct.unpack(format, embedding)

            embeddings[embeddingIndex] = embedding

            log.progress('Loading embeddings: {0:.3f}%.', embeddingIndex + 1, embeddingsCount)

        return embeddings


def loadW2VParameters(filePath, loadEmbeddings=True):
    with open(filePath, 'rb') as file:
        firstLine = file.readline()
        embeddingsCount, embeddingSize = tuple(firstLine.split(' '))
        embeddingsCount, embeddingSize = int(embeddingsCount), int(embeddingSize)
        embeddingFormat = '{0}f'.format(embeddingSize)
        wordIndexMap = {}
        embeddings = []

        log.info('Words count: {0}. Embedding size: {1}.', embeddingsCount, embeddingSize)

        embeddingIndex = 0
        while True:
            word = ''
            while True:
                char = file.read(1)

                if not char:
                    log.lineBreak()

                    if loadEmbeddings:
                        return wordIndexMap, numpy.asarray(embeddings)
                    else:
                        return wordIndexMap

                if char == ' ':
                    word = word.strip()
                    break

                word += char

            wordIndexMap[word] = len(wordIndexMap)
            if loadEmbeddings:
                embedding = struct.unpack(embeddingFormat, file.read(embeddingSize * 4))
                embeddings.append(embedding)
            else:
                file.seek(embeddingSize * 4, io.SEEK_CUR)

            embeddingIndex += 1
            log.progress('Loading W2V embeddings: {0:.3f}%.', embeddingIndex, embeddingsCount)


def pruneW2VEmbeddings(wordEmbeddingsMapPath, wordVocabularyPath, wordEmbeddingsPath, mask={}):
    wordEmbeddingsMapFile = open(wordEmbeddingsMapPath, 'r')
    wordVocabularyFile = open(wordVocabularyPath, 'w+')
    wordEmbeddingsFile = open(wordEmbeddingsPath, 'w+')

    wordsCount = 0

    try:
        firstLine = wordEmbeddingsMapFile.readline()
        embeddingsCount, embeddingSize = tuple(firstLine.split(' '))
        embeddingsCount, embeddingSize = int(embeddingsCount), int(embeddingSize)
        embeddingFormat = '{0}f'.format(embeddingSize)

        wordVocabularyFile.write(struct.pack('i', 0)) # placeholder to be filled later
        wordEmbeddingsFile.write(struct.pack('i', 0)) # placeholder to be filled later
        wordEmbeddingsFile.write(struct.pack('i', embeddingSize))

        log.info('Vocabulary size: {0}. Embedding size: {1}.', embeddingsCount, embeddingSize)
        wordIndex = 0
        while True:
            word = ''
            while True:
                char = wordEmbeddingsMapFile.read(1)

                if not char:
                    log.lineBreak()
                    return

                if char == ' ':
                    word = word.strip()
                    break

                word += char

            if word in mask:
                wordLength = len(word)

                wordVocabularyFile.write(struct.pack('i', wordLength))
                wordVocabularyFile.write(word)
                wordVocabularyFile.write(struct.pack('i', wordsCount))

                embedding = wordEmbeddingsMapFile.read(embeddingSize * 4)
                wordEmbeddingsFile.write(embedding)

                wordsCount += 1
            else:
                wordEmbeddingsMapFile.seek(embeddingSize * 4, io.SEEK_CUR)

            wordIndex += 1
            log.progress('Pruning W2V embeddings: {0:.3f}%. Pruned {1} words out of {2} ({3:.3f}%).',
                         wordIndex,
                         embeddingsCount,
                         wordIndex - wordsCount,
                         embeddingsCount,
                         (wordIndex - wordsCount) * 100.0 / embeddingsCount)
    finally:
        wordVocabularyFile.seek(0, io.SEEK_SET)
        wordEmbeddingsFile.seek(0, io.SEEK_SET)

        wordVocabularyFile.write(struct.pack('i', wordsCount))
        wordEmbeddingsFile.write(struct.pack('i', wordsCount))

        wordEmbeddingsMapFile.close()
        wordVocabularyFile.close()
        wordEmbeddingsFile.close()