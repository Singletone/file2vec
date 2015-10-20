import os
import glob
import time
import re
import struct
import io
import kit

import log
import parameters


class WordContextProvider:
    def __init__(self, textFilePath):
        self.textFile = open(textFilePath)


    def __del__(self):
        self.textFile.close()


    def next(self, contextSize, bufferSize=100):
        buffer = self.textFile.read(bufferSize)
        tail = ''

        while buffer != '':
            buffer = tail + buffer
            buffer = re.split('\.', buffer)

            tail = buffer[-1]

            for sentence in buffer[:-1]:
                words = re.split('\s+', sentence.strip())

                for wordIndex in range(len(words) - contextSize + 1):
                    window = words[wordIndex: wordIndex + contextSize]

                    yield window

            words = re.split('\s+', tail.lstrip())

            buffer = self.textFile.read(bufferSize)

            if len(words) > contextSize * 2 - 1 or buffer == '':
                if buffer != '':
                    tail = ' '.join(words[-contextSize:])
                    words = words[:-contextSize]

                for wordIndex in range(len(words) - contextSize + 1):
                    window = words[wordIndex: wordIndex + contextSize]

                    yield window


def processData(inputDirectoryPath, w2vEmbeddingsFilePath, fileIndexMapFilePath, wordIndexMapFilePath, wordEmbeddingsFilePath, contextsPath, windowSize):
    if os.path.exists(contextsPath):
        os.remove(contextsPath)

    fileContextSize = 1
    wordContextSize = windowSize - fileContextSize

    fileIndexMap = {}
    wordIndexMap = {}
    wordEmbeddings = []
    w2vWordIndexMap, w2vEmbeddings = parameters.loadW2VParameters(w2vEmbeddingsFilePath)

    if os.path.exists(contextsPath):
        os.remove(contextsPath)

    with open(contextsPath, 'wb+') as contextsFile:
        contextsFile.write(struct.pack('i', 0)) # this is a placeholder for contexts count
        contextsFile.write(struct.pack('i', windowSize))

        pathName = inputDirectoryPath + '/*/*.txt'
        textFilePaths = glob.glob(pathName)
        textFilePaths = sorted(textFilePaths)
        textFileCount = len(textFilePaths)
        startTime = time.time()

        contextFormat = '{0}i'.format(windowSize)
        contextsCount = 0

        for textFileIndex, textFilePath in enumerate(textFilePaths):
            fileIndexMap[textFilePath] = textFileIndex

            contextProvider = WordContextProvider(textFilePath)
            for wordContext in contextProvider.next(wordContextSize):
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

                contextsFile.write(struct.pack(contextFormat, *indexContext))
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

        contextsFile.seek(0, io.SEEK_SET)
        contextsFile.write(struct.pack('i', contextsCount))
        contextsFile.flush()

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
        windowSize = 8)