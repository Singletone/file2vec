import os
import glob
import time
import re
import io
import kit
import collections
import numpy

import log
import parameters
import binary


class WordContextProvider:
    def __init__(self, text=None, textFilePath=None, chunkSize=1073741824): # 1Mb as defeault buffer size
        self.text = text
        self.textFile = None
        self.chunkSize = chunkSize

        if textFilePath is not None:
            self.textFile = open(textFilePath)


    def __del__(self):
        if self.textFile is not None:
            self.textFile.close()


    def iterate(self, contextSize):
        if self.textFile is not None:
            chunk = self.textFile.read(self.chunkSize)
        else:
            chunk = self.text

        tail = ''

        while chunk != '':
            chunk = tail + chunk
            chunk = re.split('\.', chunk)

            tail = chunk[-1]

            for sentence in chunk[:-1]:
                sentence = sentence.strip()
                words = re.split('\s+', sentence)

                for wordIndex in range(len(words) - contextSize + 1):
                    window = words[wordIndex: wordIndex + contextSize]

                    yield window

            words = re.split('\s+', tail.lstrip())

            if self.textFile is not None:
                chunk = self.textFile.read(self.chunkSize)
            else:
                chunk = ''

            if len(words) > contextSize * 2 - 1 or chunk == '':
                if chunk != '':
                    tail = ' '.join(words[-contextSize:])
                    words = words[:-contextSize]

                for wordIndex in range(len(words) - contextSize + 1):
                    window = words[wordIndex: wordIndex + contextSize]

                    yield window


def generateNegativeSamples(negative, contexts, wordIndexMap, strict=False):
    wordIndices = map(lambda item: item[1], wordIndexMap.items())
    wordIndices = numpy.asarray(wordIndices)
    highWordIndex = max(wordIndices)

    if strict:
        negativeSamples = numpy.zeros((contexts.shape[0], negative))
        for contextIndex in xrange(0, contexts.shape[0]):
            prunedWordIndices = wordIndices[wordIndices != contexts[contextIndex, -1]]
            numpy.random.shuffle(prunedWordIndices)
            negativeSamples[contextIndex] = prunedWordIndices[:negative]
    else:
        negativeSamples = numpy.random.randint(0, high=highWordIndex+1, size=contexts.shape[0]*negative)
        negativeSamples = negativeSamples.reshape((contexts.shape[0], negative))

    return negativeSamples


def processData(inputDirectoryPath, w2vEmbeddingsFilePath, fileIndexMapFilePath,
                wordIndexMapFilePath, wordEmbeddingsFilePath, contextsPath, windowSize, negative, strict):
    if os.path.exists(contextsPath):
        os.remove(contextsPath)

    fileContextSize = 1
    wordContextSize = windowSize - fileContextSize

    fileIndexMap = {}
    wordIndexMap = collections.OrderedDict()
    wordEmbeddings = []

    noNegativeSamplingPath = contextsPath
    if negative > 0:
        noNegativeSamplingPath += '.temp'

    if os.path.exists(noNegativeSamplingPath):
        os.remove(noNegativeSamplingPath)

    pathName = inputDirectoryPath + '/*.txt'
    textFilePaths = glob.glob(pathName)
    textFilePaths = sorted(textFilePaths)
    textFileCount = len(textFilePaths)

    w2vWordIndexMap, w2vEmbeddings = parameters.loadW2VParameters(w2vEmbeddingsFilePath)

    contextsCount = 0
    with open(noNegativeSamplingPath, 'wb+') as noNegativeSamplingFile:
        binary.writei(noNegativeSamplingFile, 0) # this is a placeholder for contexts count
        binary.writei(noNegativeSamplingFile, windowSize)
        binary.writei(noNegativeSamplingFile, 0)

        startTime = time.time()

        for textFileIndex, textFilePath in enumerate(textFilePaths):
            fileIndexMap[textFilePath] = textFileIndex

            contextProvider = WordContextProvider(textFilePath=textFilePath)
            for wordContext in contextProvider.iterate(wordContextSize):
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

                binary.writei(noNegativeSamplingFile, indexContext)
                contextsCount += 1

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
        binary.writei(noNegativeSamplingFile, contextsCount)
        noNegativeSamplingFile.flush()

    if negative > 0:
        with open(contextsPath, 'wb+') as contextsFile:
            startTime = time.time()

            contextProvider = parameters.IndexContextProvider(noNegativeSamplingPath)

            binary.writei(contextsFile, contextsCount)
            binary.writei(contextsFile, windowSize)
            binary.writei(contextsFile, negative)

            batchSize = 10000
            batchesCount = contextsCount / batchSize + 1

            for batchIndex in xrange(0, batchesCount):
                contexts = contextProvider[batchIndex * batchSize : (batchIndex + 1) * batchSize]
                negativeSamples = generateNegativeSamples(negative, contexts, wordIndexMap, strict)
                contexts = numpy.concatenate([contexts, negativeSamples], axis=1)
                contexts = numpy.ravel(contexts)

                binary.writei(contextsFile, contexts)

                currentTime = time.time()
                elapsed = currentTime - startTime

                log.progress('Negative sampling: {0:.3f}%. Elapsed: {1}.',
                     batchIndex + 1,
                     batchesCount,
                     log.delta(elapsed))

            log.lineBreak()
            contextsFile.flush()

            os.remove(noNegativeSamplingPath)

    parameters.dumpIndexMap(fileIndexMap, fileIndexMapFilePath)
    parameters.dumpIndexMap(wordIndexMap, wordIndexMapFilePath)
    parameters.dumpEmbeddings(wordEmbeddings, wordEmbeddingsFilePath)


def launch(pathTo, hyper):
    processData(
        inputDirectoryPath = pathTo.weededDir,
        w2vEmbeddingsFilePath = pathTo.w2vEmbeddings,
        fileIndexMapFilePath = pathTo.fileIndexMap,
        wordIndexMapFilePath = pathTo.wordIndexMap,
        wordEmbeddingsFilePath = pathTo.wordEmbeddings,
        contextsPath = pathTo.contexts,
        windowSize = hyper.windowSize,
        negative = hyper.negative,
        strict = hyper.strict)


if __name__ == '__main__':
    pathTo = kit.PathTo('Wikipedia', 'wiki_full_s800_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(windowSize=7, negative=100, strict=False)

    launch(pathTo, hyper)