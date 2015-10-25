import os
import glob
import time
import re
import struct
import io
import kit
import collections
import numpy

import log
import parameters


class WordContextProvider:
    def __init__(self, textFilePath, bufferSize=1073741824): # 1Mb as defeault buffer size
        self.textFile = open(textFilePath)
        self.bufferSize = bufferSize


    def __del__(self):
        self.textFile.close()


    def next(self, contextSize):
        buffer = self.textFile.read(self.bufferSize)
        tail = ''

        while buffer != '':
            buffer = tail + buffer
            buffer = re.split('\.', buffer)

            tail = buffer[-1]

            for sentence in buffer[:-1]:
                sentence = sentence.strip()
                words = re.split('\s+', sentence)

                for wordIndex in range(len(words) - contextSize + 1):
                    window = words[wordIndex: wordIndex + contextSize]

                    yield sentence, window

            words = re.split('\s+', tail.lstrip())

            buffer = self.textFile.read(self.bufferSize)

            if len(words) > contextSize * 2 - 1 or buffer == '':
                if buffer != '':
                    tail = ' '.join(words[-contextSize:])
                    words = words[:-contextSize]

                for wordIndex in range(len(words) - contextSize + 1):
                    window = words[wordIndex: wordIndex + contextSize]

                    yield tail, window


def getWordFrequencyMap(text):
    words = re.split('\s+', text)
    frequencies = [(word, text.count(word)) for word in words]
    wordFrequencyMap = collections.OrderedDict(frequencies)

    return wordFrequencyMap


def mergeFrequencyMaps(*frequencyMaps):
    mergedMap = collections.OrderedDict()

    for frequencyMap in frequencyMaps:
        for word, frequency in frequencyMap.items():
            if word in mergedMap:
                mergedMap[word] += frequency
            else:
                mergedMap[word] = frequency

    return mergedMap


def generateNegativeSample(negative, context, wordIndexMap, wordFrequencyMap):
    wordIndices = map(lambda item: item[1], wordIndexMap.items())
    wordIndices = [index for index in wordIndices if index != context[-1]]
    numpy.random.shuffle(wordIndices)

    return wordIndices[:negative]


def processData(inputDirectoryPath, w2vEmbeddingsFilePath, fileIndexMapFilePath, wordIndexMapFilePath, wordEmbeddingsFilePath, contextsPath, windowSize, negative):
    if os.path.exists(contextsPath):
        os.remove(contextsPath)

    fileContextSize = 1
    wordContextSize = windowSize - fileContextSize

    fileIndexMap = {}
    wordIndexMap = collections.OrderedDict()
    wordFrequencyMap = collections.OrderedDict()
    wordEmbeddings = []
    w2vWordIndexMap, w2vEmbeddings = parameters.loadW2VParameters(w2vEmbeddingsFilePath)

    noNegativeSamplingPath = contextsPath
    if negative > 0:
        noNegativeSamplingPath += '.temp'

    if os.path.exists(noNegativeSamplingPath):
        os.remove(noNegativeSamplingPath)

    contextsCount = 0
    contextsFormat = '{0}i'.format(windowSize)
    with open(noNegativeSamplingPath, 'wb+') as noNegativeSamplingFile:
        noNegativeSamplingFile.write(struct.pack('i', 0)) # this is a placeholder for contexts count
        noNegativeSamplingFile.write(struct.pack('i', windowSize))
        noNegativeSamplingFile.write(struct.pack('i', 0))

        pathName = inputDirectoryPath + '/*/*.txt'
        textFilePaths = glob.glob(pathName)
        textFilePaths = sorted(textFilePaths)
        textFileCount = len(textFilePaths)
        startTime = time.time()

        for textFileIndex, textFilePath in enumerate(textFilePaths):
            fileIndexMap[textFilePath] = textFileIndex

            currentSentence = None
            contextProvider = WordContextProvider(textFilePath)
            for sentence, wordContext in contextProvider.next(wordContextSize):
                if currentSentence != sentence:
                    currentSentence = sentence
                    frequencies = getWordFrequencyMap(currentSentence)
                    wordFrequencyMap = mergeFrequencyMaps(wordFrequencyMap, frequencies)

                allWordsInWordVocabulary = [word in w2vWordIndexMap for word in wordContext]

                if not all(allWordsInWordVocabulary):
                    continue

                for word in wordContext:
                    if word not in wordIndexMap:
                        wordIndexMap[word] = len(wordIndexMap)
                        wordEmbeddingIndex = w2vWordIndexMap[word]
                        wordEmbedding = w2vEmbeddings[wordEmbeddingIndex]
                        wordEmbeddings.append(wordEmbedding)

                indexContext = [textFileIndex] + map(lambda w: wordIndexMap[w], wordContext)

                noNegativeSamplingFile.write(struct.pack(contextsFormat, *indexContext))
                contextsCount += 1

            textFileName = os.path.basename(textFilePath)
            currentTime = time.time()
            elapsed = currentTime - startTime
            secondsPerFile = elapsed / (textFileIndex + 1)

            log.progress('Reading contexts: {0:.3f}%. Elapsed: {1} ({2:.3f} sec/file). Words: {3}. Contexts: {4}.',
                         textFileIndex + 1,
                         textFileCount,
                         log.delta(elapsed),
                         secondsPerFile,
                         len(wordIndexMap),
                         contextsCount)

        log.lineBreak()

        noNegativeSamplingFile.seek(0, io.SEEK_SET)
        noNegativeSamplingFile.write(struct.pack('i', contextsCount))
        noNegativeSamplingFile.flush()

    if negative > 0:
        contextsFormat = '{0}i'.format(windowSize + negative)

        with open(contextsPath, 'wb+') as contextsFile:
            startTime = time.time()

            contextProvider = parameters.IndexContextProvider(noNegativeSamplingPath)

            contextsFile.write(struct.pack('i', contextsCount))
            contextsFile.write(struct.pack('i', windowSize))
            contextsFile.write(struct.pack('i', negative))

            for contextIndex in xrange(0, contextsCount):
                context = contextProvider[contextIndex]
                negativeSample = generateNegativeSample(negative, context, wordIndexMap, wordFrequencyMap)
                context = numpy.concatenate([context, negativeSample])

                contextsFile.write(struct.pack(contextsFormat, *context))

                currentTime = time.time()
                elapsed = currentTime - startTime

                log.progress('Random sampling: {0:.3f}%. Elapsed: {1}.',
                     contextIndex + 1,
                     contextsCount,
                     log.delta(elapsed))

            log.lineBreak()
            contextsFile.flush()

            os.remove(noNegativeSamplingPath)

    parameters.dumpIndexMap(fileIndexMap, fileIndexMapFilePath)
    parameters.dumpIndexMap(wordIndexMap, wordIndexMapFilePath)
    parameters.dumpEmbeddings(wordEmbeddings, wordEmbeddingsFilePath)


if __name__ == '__main__':
    pathTo = kit.PathTo('Duplicates')

    processData(
        inputDirectoryPath = pathTo.preparedDir,
        w2vEmbeddingsFilePath = pathTo.w2vEmbeddings('wiki_full_s100_w7_n7.bin'),
        fileIndexMapFilePath = pathTo.fileIndexMap,
        wordIndexMapFilePath = pathTo.wordIndexMap,
        wordEmbeddingsFilePath = pathTo.wordEmbeddings,
        contextsPath = pathTo.contexts,
        windowSize = 8,
        negative = 10)