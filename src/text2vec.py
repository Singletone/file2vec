import collections
from os.path import exists
import gc
import time
import numpy

import connectors
import extraction
import kit
import log
import parameters
import processing
import weeding
import binary


batchSize = 10000


def extract(connector):
    textFilesCount = connector.count()

    names = []
    texts = []
    for textFileIndex, name, text in connector.iterate():
        text = extraction.clean(text)

        names.append(name)
        texts.append(text)

        log.progress('Extracting text: {0:.3f}%. Texts: {1}.', textFileIndex + 1, textFilesCount, textFileIndex + 1)

    log.lineBreak()

    return names, texts


def buildWordMaps(texts, w2vWordIndexMap, w2vWordEmbeddings):
    wordIndexMap = collections.OrderedDict()
    wordFrequencyMap = collections.OrderedDict()

    for textIndex, text in enumerate(texts):
        for word in weeding.iterateWords(text):
            if word not in w2vWordIndexMap:
                continue

            if word not in wordIndexMap:
                wordIndexMap[word] = len(wordIndexMap)
                wordFrequencyMap[word] = 1
            else:
                wordFrequencyMap[word] += 1

        log.progress('Building word maps: {0:.3f}%. Words: {1}.', textIndex + 1, len(texts), len(wordIndexMap))

    log.lineBreak()

    wordEmbeddings = numpy.zeros((len(wordIndexMap), w2vWordEmbeddings.shape[1]))
    for wirdIndexPair in wordIndexMap.items():
        word, index = wirdIndexPair
        wordEmbeddings[index] = w2vWordEmbeddings[index]

        log.progress('Copying word w2v embeddings: {0:.3f}%.', index + 1, len(wordIndexMap))

    log.lineBreak()

    return wordIndexMap, wordFrequencyMap, wordEmbeddings


def subsampleAndPrune(texts, wordFrequencyMap, sample, minCount):
    totalLength = 0.
    prunedLength = 0.

    for textIndex, text in enumerate(texts):
        totalLength += len(text)

        texts[textIndex] = weeding.subsampleAndPrune(text, wordFrequencyMap, sample, minCount)

        prunedLength += len(texts[textIndex])

        log.progress('Subsampling and pruning text: {0:.3f}%. Removed {1:.3f}% of original text.',
                     textIndex + 1,
                     len(texts),
                     (1 - prunedLength/totalLength) * 100)

    log.lineBreak()

    return texts


def inferContexts(texts, wordIndexMap, windowSize, negative, strict):
    contexts = []

    def wordsToIndices(textContext):
        indices = map(lambda word: wordIndexMap[word], textContext)
        return indices

    print 'Reading contexts:'

    for textIndex, text in enumerate(texts):
        print '-----------'
        print text

        contextProvider = processing.WordContextProvider(text=text, minContexts=600)
        textContexts = list(contextProvider.iterate(windowSize))

        textContexts = map(wordsToIndices, textContexts)
        contexts.append(textContexts)

        # log.progress('Creating contexts: {0:.3f}%. Current text: {1}.', textIndex + 1, len(texts))

    log.lineBreak()

    return contexts


def trainTextVectors(connector, w2vEmbeddingsPath, wordIndexMapPath, wordFrequencyMapPath, wordEmbeddingsPath, contextsPath,
                     sample, minCount, windowSize, negative, strict):
    names = []
    texts = []

    if exists(wordIndexMapPath) and exists(wordFrequencyMapPath) and exists(wordEmbeddingsPath):
        wordIndexMap = parameters.loadWordMap(wordIndexMapPath)
        wordFrequencyMap = parameters.loadWordMap(wordFrequencyMapPath)
        wordEmbeddings = parameters.loadEmbeddings(wordEmbeddingsPath)

        log.info('Loaded indices, frequencies and embeddings')
    else:
        w2vWordIndexMap, w2vWordEmbeddings = parameters.loadW2VParameters(w2vEmbeddingsPath)

        names, texts = extract(connector)
        wordIndexMap, wordFrequencyMap, wordEmbeddings = buildWordMaps(texts, w2vWordIndexMap, w2vWordEmbeddings)

        # parameters.dumpWordMap(wordIndexMap, wordIndexMapPath)
        del w2vWordIndexMap
        del w2vWordEmbeddings
        gc.collect()

        parameters.dumpWordMap(wordFrequencyMap, wordFrequencyMapPath)

        # parameters.dumpEmbeddings(wordEmbeddings, wordEmbeddingsPath)

        log.info('Dumped indices, frequencies and embeddings')

    if exists(contextsPath):
        contexts = binary.loadTensor(contextsPath)

        log.info('Loaded contexts')
    else:
        texts = subsampleAndPrune(texts, wordFrequencyMap, sample, minCount)

        contexts = inferContexts(texts, wordIndexMap, windowSize, negative, strict)

        log.progress('Dumping contexts...', 1, 1)
        binary.dumpTensor(contextsPath, contexts)
        log.info('Dumping contexts complete.')


def launch(pathTo, hyper):
    w2vEmbeddingsPath = pathTo.w2vEmbeddings
    contextsPath = pathTo.contexts
    wordIndexMap = pathTo.wordIndexMap
    wordFrequencyMap = pathTo.wordFrequencyMap
    wordEmbeddings = pathTo.wordEmbeddings

    connector = hyper.connector
    windowSize = hyper.windowSize
    negative = hyper.negative
    strict = hyper.strict

    trainTextVectors(connector, w2vEmbeddingsPath, wordIndexMap, wordFrequencyMap, wordEmbeddings, contextsPath,
                     hyper.sample, hyper.minCount, windowSize, negative, strict)


if __name__ == '__main__':
    pathTo = kit.PathTo('IMDB', experiment='imdb', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(
        connector = connectors.ImdbConnector(pathTo.dataSetDir),
        sample=2e1,
        minCount=1,
        windowSize=3,
        negative=100,
        strict=False,
        fileEmbeddingSize=1000,
        epochs=10,
        batchSize=1,
        learningRate=0.025
    )

    launch(pathTo, hyper)