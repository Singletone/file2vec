import collections
from os.path import exists
import gc
import numpy
import io

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
    for wordIndexPair in wordIndexMap.items():
        word, index = wordIndexPair
        wordEmbeddings[index] = w2vWordEmbeddings[index]

        log.progress('Copying w2v embeddings: {0:.3f}%.', index + 1, len(wordIndexMap))

    log.lineBreak()

    return wordIndexMap, wordFrequencyMap, wordEmbeddings


def subsampleAndPrune(texts, wordFrequencyMap, sample, minCount):
    totalLength = 0.
    prunedLength = 0.

    maxFrequency = wordFrequencyMap.items()[0][1]

    for textIndex, text in enumerate(texts):
        totalLength += len(text)

        texts[textIndex] = weeding.subsampleAndPrune(text, wordFrequencyMap, maxFrequency, sample, minCount)

        prunedLength += len(texts[textIndex])

        log.progress('Subsampling and pruning text: {0:.3f}%. Removed {1:.3f}% of original text.',
                     textIndex + 1,
                     len(texts),
                     (1 - prunedLength/totalLength) * 100)

    log.lineBreak()

    return texts


def inferContexts(contextsPath, names, texts, wordIndexMap, windowSize, negative, strict, minContexts, maxContexts):
    textIndexMap = collections.OrderedDict()

    def wordsToIndices(textContext):
        indices = map(lambda word: wordIndexMap[word], textContext)
        return indices

    wordIndices = map(lambda item: item[1], wordIndexMap.items())
    wordIndices = numpy.asarray(wordIndices)
    maxWordIndex = max(wordIndices)

    with open(contextsPath, 'wb+') as contextsFile:
        binary.writei(contextsFile, 0)
        binary.writei(contextsFile, windowSize)
        binary.writei(contextsFile, negative)

        textIndex = 0
        contextsCount = 0
        for name, text in zip(names, texts):
            contextProvider = processing.WordContextProvider(text=text, minContexts=minContexts, maxContexts=maxContexts)
            contexts = list(contextProvider.iterate(windowSize))

            if len(contexts) > 0:
                contexts = map(wordsToIndices, contexts)
                textIndexMap[name] = len(textIndexMap)
                contexts = numpy.asarray(contexts)
                contextsCount += len(contexts)

                contexts = processing.generateNegativeSamples(negative, contexts, wordIndices, maxWordIndex, strict)
                contexts = numpy.ravel(contexts)

                binary.writei(contextsFile, contexts)

            textIndex += 1
            log.progress('Creating contexts: {0:.3f}%. Text index map: {1}. Contexts: {2}.',
                         textIndex, len(texts), len(textIndexMap), contextsCount)

        contextsFile.seek(0, io.SEEK_SET)
        binary.writei(contextsFile, contextsCount)
        contextsFile.flush()

    log.lineBreak()

    return textIndexMap


def trainTextVectors(connector, w2vEmbeddingsPath, wordIndexMapPath, wordFrequencyMapPath, wordEmbeddingsPath, contextsPath,
                     sample, minCount, windowSize, negative, strict):
    if exists(wordIndexMapPath) and exists(wordFrequencyMapPath) and exists(wordEmbeddingsPath) \
            and exists(contextsPath) and exists(pathTo.textIndexMap):
        wordIndexMap = parameters.loadMap(wordIndexMapPath)
        wordFrequencyMap = parameters.loadMap(wordFrequencyMapPath)
        wordEmbeddings = parameters.loadEmbeddings(wordEmbeddingsPath)
        textIndexMap = parameters.loadMap(pathTo.textIndexMap)

        log.progress('Loading contexts...')
        contexts = binary.loadTensor(contextsPath)
        log.info('Loaded {0} contexts.', len(contexts))
    else:
        w2vWordIndexMap, w2vWordEmbeddings = parameters.loadW2VParameters(w2vEmbeddingsPath)

        names, texts = extract(connector)
        wordIndexMap, wordFrequencyMap, wordEmbeddings = buildWordMaps(texts, w2vWordIndexMap, w2vWordEmbeddings)

        parameters.dumpWordMap(wordIndexMap, wordIndexMapPath)
        del w2vWordIndexMap
        del w2vWordEmbeddings
        gc.collect()

        parameters.dumpWordMap(wordFrequencyMap, wordFrequencyMapPath)

        log.progress('Dumping contexts...')
        parameters.dumpEmbeddings(wordEmbeddings, wordEmbeddingsPath)
        log.info('Dumped indices, frequencies and embeddings')

        texts = subsampleAndPrune(texts, wordFrequencyMap, sample, minCount)

        textIndexMap = inferContexts(contextsPath, names, texts, wordIndexMap, windowSize, negative, strict, 600, 600)

        parameters.dumpWordMap(textIndexMap, pathTo.textIndexMap)



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
    pathTo = kit.PathTo('IMDB', experiment='imdb', w2vEmbeddings='mojito_s300_w3_mc5_hs1.bin')
    hyper = parameters.HyperParameters(
        connector = connectors.ImdbConnector(pathTo.dataSetDir),
        threshold=1e-3,
        minCount=1,
        windowSize=3,
        negative=10,
        strict=False,
        fileEmbeddingSize=1000,
        epochs=10,
        batchSize=1,
        learningRate=0.025
    )

    launch(pathTo, hyper)