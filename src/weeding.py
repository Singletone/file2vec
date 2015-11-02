import numpy
import collections
import re
import glob
import math
import os
import shutil

import kit
import log


def iterateSentences(text):
    sentences = re.split('\.', text)
    for sentence in sentences:
        if sentence != '':
            yield sentence


def iterateWords(text):
    for sentence in iterateSentences(text):
        words = re.split('\s+', sentence)

        for word in words:
            if word != '':
                yield word


def subsampleAndPrune(text, wordFrequencyMap, sample, minCount):
    sentences = []

    for sentence in iterateSentences(text):
        words = []

        for word in iterateWords(sentence):
            wordFrequency = wordFrequencyMap[word]
            wordSample = numpy.random.random()

            if word in wordFrequencyMap and wordSample > (1 - math.sqrt(sample/wordFrequency)) and wordFrequency >= minCount:
                words.append(word)

        if len(words) > 0:
            sentence = ' '.join(words)
            sentences.append(sentence)

    text = ''
    if len(sentences) > 0:
        text = '. '.join(sentences) + '.'

    return text


def weed(inputDirectoryPath, outputDirectoryPath, sample, minCount):
    if os.path.exists(outputDirectoryPath):
        shutil.rmtree(outputDirectoryPath, ignore_errors=True)

    os.mkdir(outputDirectoryPath)
    os.chown(outputDirectoryPath, 1000, 1000)

    pathName = inputDirectoryPath + '/*.txt'
    textFilePaths = glob.glob(pathName)
    textFilePaths = sorted(textFilePaths)
    textFilesCount = len(textFilePaths)

    wordFrequencyMap = collections.OrderedDict()

    for textFileIndex, textFilePath in enumerate(textFilePaths):
        with open(textFilePath, 'r') as textFile:
            text = textFile.read()

            for word in iterateWords(text):
                if word not in wordFrequencyMap:
                    wordFrequencyMap[word] = 1
                else:
                    wordFrequencyMap[word] += 1

        log.progress('Building frequency map: {0:.3f}.', textFileIndex + 1, textFilesCount)

    log.lineBreak()

    wordFrequencyMap = sorted(wordFrequencyMap.items(), key=lambda item: item[1], reverse=True)
    wordFrequencyMap = collections.OrderedDict(wordFrequencyMap)

    for textFileIndex, textFilePath in enumerate(textFilePaths):
        with open(textFilePath, 'r') as textFile:
            text = textFile.read()
            text = subsampleAndPrune(text, wordFrequencyMap, sample, minCount)

            fileName = os.path.basename(textFilePath)
            weededFilePath = os.path.join(outputDirectoryPath, fileName)

            with open(weededFilePath, 'w+') as weededFile:
                weededFile.write(text)

        log.progress('Pruning and subsampling: {0:.3f}.', textFileIndex + 1, textFilesCount)

    log.lineBreak()


def launch(pathTo):
    weed(
        inputDirectoryPath = pathTo.extractedDir,
        outputDirectoryPath = pathTo.weededDir,
        sample = 1e1,
        minCount = 5
    )

if __name__ == '__main__':
    pathTo = kit.PathTo('Cockatoo')
    launch(pathTo)

