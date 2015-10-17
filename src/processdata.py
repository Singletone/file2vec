import os
import glob
import log
import time
import re
import struct
import io
import collections
import math
import whitelist
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


def processData(inputDirectoryPath, wordEmbeddingsMapPath, fileVocabularyPath, wordVocabularyPath, wordEmbeddingsPath, contextsPath, contextSize):
    if os.path.exists(contextsPath):
        os.remove(contextsPath)

    fileContextSize = 1
    wordContextSize = contextSize - fileContextSize

    fileVocabulary = collections.OrderedDict()
    wordIndexMap = collections.OrderedDict()
    w2vWordIndexMap = parameters.loadW2VParameters(wordEmbeddingsMapPath, loadEmbeddings=False)

    if os.path.exists(contextsPath):
        os.remove(contextsPath)

    with open(contextsPath, 'wb+') as contextsFile:
        contextsFile.write(struct.pack('i', 0)) # this is a placeholder for contexts count
        contextsFile.write(struct.pack('i', contextSize))

        pathName = inputDirectoryPath + '/*/*.txt'
        textFilePaths = glob.glob(pathName)
        textFilePaths = sorted(textFilePaths)
        textFileCount = len(textFilePaths)
        startTime = time.time()

        contextFormat = '{0}i'.format(contextSize)
        contextsCount = 0

        for textFileIndex, textFilePath in enumerate(textFilePaths):
            fileVocabulary[textFilePath] = textFileIndex

            contextProvider = WordContextProvider(textFilePath)
            for wordContext in contextProvider.next(wordContextSize):
                allWordsInWordVocabulary = [word in w2vWordIndexMap for word in wordContext]

                if not all(allWordsInWordVocabulary):
                    continue

                for word in wordContext:
                    if word not in wordIndexMap:
                        wordIndexMap[word] = len(wordIndexMap)

                indexContext = [textFileIndex] + map(lambda w: w2vWordIndexMap[w], wordContext)

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

    parameters.dumpFileVocabulary(fileVocabulary, fileVocabularyPath)
    parameters.pruneW2VEmbeddings(wordEmbeddingsMapPath, wordVocabularyPath, wordEmbeddingsPath, mask=wordIndexMap)


if __name__ == '__main__':
    inputDirectoryPath = '../data/Drosophila/Prepared'
    wordEmbeddingsMapPath = '../data/Drosophila/Processed/drosophila_w2v.bin'
    fileVocabularyPath = '../data/Drosophila/Parameters/file_vocabulary.bin'
    wordVocabularyPath = '../data/Drosophila/Parameters/word_vocabulary.bin'
    wordEmbeddingsPath = '../data/Drosophila/Parameters/word_embeddings.bin'
    contextsPath = '../data/Drosophila/Processed/contexts.bin'
    contextSize = 7

    processData(inputDirectoryPath, wordEmbeddingsMapPath, fileVocabularyPath, wordVocabularyPath, wordEmbeddingsPath, contextsPath, contextSize)