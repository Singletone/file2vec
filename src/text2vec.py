import collections

import connectors
import extraction
import kit
import log
import parameters
import weeding


batchSize = 10000


def extract(connector):
    textFilesCount = connector.count()

    names = []
    texts = []
    for textFileIndex, name, text in connector.iterate():
        text = extraction.clean(text)

        names.append(name)
        texts.append(text)

        log.progress('Extracting text: {0:.3f}%.', textFileIndex + 1, textFilesCount)

    log.lineBreak()

    return names, texts


def buildFrequencyMap(texts):
    wordFrequencyMap = collections.OrderedDict()

    for textIndex, text in enumerate(texts):
        for word in weeding.iterateWords(text):
            if word not in wordFrequencyMap:
                wordFrequencyMap[word] = 1
            else:
                wordFrequencyMap[word] += 1

        log.progress('Building frequency map: {0:.3f}%.', textIndex + 1, len(texts))

    log.lineBreak()

    return wordFrequencyMap


def subsampleAndPrune(texts, wordFrequencyMap, sample, minCount):
    for textIndex, text in enumerate(texts):
        texts[textIndex] = weeding.subsampleAndPrune(text, wordFrequencyMap, sample, minCount)

        log.progress('Subsampling and pruning text: {0:.3f}%.', textIndex + 1, len(texts))

    log.lineBreak()

    return texts


def trainTextVectors(connector, sample, minCount):
    names, texts = extract(connector)
    wordFrequencyMap = buildFrequencyMap(texts)
    texts = subsampleAndPrune(texts, wordFrequencyMap, sample, minCount)


def launch(pathTo, hyper):
    connector = hyper.connector

    trainTextVectors(connector, hyper.sample, hyper.minCount)


if __name__ == '__main__':
    pathTo = kit.PathTo('IMDB', experiment='imdb', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(
        sample=1e1,
        minCount=2,
        connector = connectors.ImdbConnector(pathTo.dataSetDir))

    launch(pathTo, hyper)