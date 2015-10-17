import os
import io
import log
import struct
import collections
import numpy


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


class EmbeddingsProvider():
    def __init__(self, embeddingsFilePath):
        self.embeddingsFilePath = embeddingsFilePath

    def getEmbedding(self):
        with open(self.embeddingsFilePath, 'r') as embeddingsFile:
            embedding = embeddingsFile.readall()
            return embedding


def dumpFileVocabulary(vocabulary, vocabularyFilePath):
    if os.path.exists(vocabularyFilePath):
        os.remove(vocabularyFilePath)

    itemsCount = len(vocabulary)
    itemIndex = 0

    with open(vocabularyFilePath, 'w') as file:
        file.write(struct.pack('i', itemsCount))

        for key, index in vocabulary.items():
            keyLength = len(key)
            keyLength = struct.pack('i', keyLength)
            index = struct.pack('i', index)

            file.write(keyLength)
            file.write(key)
            file.write(index)

            itemIndex += 1
            log.progress('Dumping file vocabulary: {0:.3f}%.', itemIndex, itemsCount)

        file.flush()

        log.lineBreak()


def loadFileVocabulary(vocabularyFilePath):
    vocabulary = collections.OrderedDict()

    with open(vocabularyFilePath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        for itemIndex in range(0, itemsCount):
            wordLength = file.read(4)
            wordLength = struct.unpack('i', wordLength)[0]

            word = file.read(wordLength)

            index = file.read(4)
            index = struct.unpack('i', index)[0]

            vocabulary[word] = index

            log.progress('Loading file vocabulary: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.lineBreak()

    return vocabulary


def loadIndexFileMap(vocabularyFilePath):
    with open(vocabularyFilePath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        indexFileMap = [None] * itemsCount

        for itemIndex in range(0, itemsCount):
            wordLength = file.read(4)
            wordLength = struct.unpack('i', wordLength)[0]

            word = file.read(wordLength)

            index = file.read(4)
            index = struct.unpack('i', index)[0]

            indexFileMap[index] = word

            log.progress('Loading index-file map: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.lineBreak()

        return indexFileMap


def getFileVocabularySize(fileVocabularyPath):
    with open(fileVocabularyPath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        return itemsCount


def dumpWordVocabulary(vocabulary, vocabularyFilePath, dumpFrequency=True):
    if os.path.exists(vocabularyFilePath):
        os.remove(vocabularyFilePath)

    itemsCount = len(vocabulary)
    itemIndex = 0

    with open(vocabularyFilePath, 'w') as file:
        file.write(struct.pack('i', itemsCount))

        for key, value in vocabulary.items():
            keyLength = len(key)
            keyLength = struct.pack('i', keyLength)

            if dumpFrequency:
                index, frequency = value
                frequency = struct.pack('i', frequency)
            else:
                index = value

            index = struct.pack('i', index)

            file.write(keyLength)
            file.write(key)
            file.write(index)

            if dumpFrequency:
                file.write(frequency)

            itemIndex += 1
            log.progress('Dumping word vocabulary: {0:.3f}%.', itemIndex, itemsCount)

        file.flush()

        log.lineBreak()


def loadWordVocabulary(vocabularyFilePath):
    vocabulary = collections.OrderedDict()

    with open(vocabularyFilePath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        for itemIndex in range(0, itemsCount):
            wordLength = file.read(4)
            wordLength = struct.unpack('i', wordLength)[0]

            word = file.read(wordLength)

            index = file.read(4)
            index = struct.unpack('i', index)[0]

            vocabulary[word] = index

            log.progress('Loading word vocabulary: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.lineBreak()

    return vocabulary


def loadIndexWordMap(vocabularyFilePath):
    with open(vocabularyFilePath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        indexWordMap = [None] * itemsCount

        for itemIndex in range(0, itemsCount):
            wordLength = file.read(4)
            wordLength = struct.unpack('i', wordLength)[0]

            word = file.read(wordLength)

            index = file.read(4)
            index = struct.unpack('i', index)[0]

            indexWordMap[index] = word

            log.progress('Loading index-word map: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.lineBreak()

        return indexWordMap


def getWordVocabularySize(wordVocabularyPath):
    with open(wordVocabularyPath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        return itemsCount


def dumpEmbeddings(embeddings, embeddingsFilePath):
    if os.path.exists(embeddingsFilePath):
        os.remove(embeddingsFilePath)

    wordsCount, embeddingSize = embeddings.shape

    with open(embeddingsFilePath, 'w') as file:
        file.write(struct.pack('<i', wordsCount))
        file.write(struct.pack('<i', embeddingSize))

        format = '{0}f'.format(embeddingSize)
        for wordIndex in range(0, wordsCount):
            wordEmbedding = embeddings[wordIndex]
            wordEmbedding = struct.pack(format, *wordEmbedding)

            file.write(wordEmbedding)


def loadEmbeddigns(embeddingsFilePath):
    with open(embeddingsFilePath, 'rb') as file:
        wordsCount = file.read(4)
        wordsCount = struct.unpack('<i', wordsCount)[0]

        embeddingSize = file.read(4)
        embeddingSize = struct.unpack('<i', embeddingSize)[0]

        embeddings = numpy.empty((wordsCount, embeddingSize))

        format = '{0}f'.format(embeddingSize)
        for wordIndex in range(0, wordsCount):
            wordEmbedding = file.read(embeddingSize * 4)
            wordEmbedding = struct.unpack(format, wordEmbedding)[0]

        return embeddings


def loadW2VParameters(filePath, loadEmbeddings=False):
    with open(filePath, 'rb') as file:
        firstLine = file.readline()
        embeddingsCount, embeddingSize = tuple(firstLine.split(' '))
        embeddingsCount, embeddingSize = int(embeddingsCount), int(embeddingSize)
        embeddingFormat = '{0}f'.format(embeddingSize)
        wordIndexMap = {}
        embeddings = []

        log.info('Vocabulary size: {0}. Embedding size: {1}.', embeddingsCount, embeddingSize)

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
            log.progress('Reading embeddings: {0:.3f}%.', embeddingIndex, embeddingsCount)


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
                         (wordIndex - wordsCount) * 100 / embeddingsCount)
    finally:
        wordVocabularyFile.seek(0, io.SEEK_SET)
        wordEmbeddingsFile.seek(0, io.SEEK_SET)

        wordVocabularyFile.write(struct.pack('i', wordsCount))
        wordEmbeddingsFile.write(struct.pack('i', wordsCount))

        wordEmbeddingsMapFile.close()
        wordVocabularyFile.close()
        wordEmbeddingsFile.close()