import log
import kit
import math
import scipy.stats
import numpy
import pandas
import re
import vectors
import pandas
import os
import random
from matplotlib import colors
from matplotlib import pyplot as plt


rubensteinGoodenoughData = None
def rubensteinGoodenough(wordIndexMap, embeddings):
    global rubensteinGoodenoughData

    if rubensteinGoodenoughData is None:
        rubensteinGoodenoughData = []

        rubensteinGoodenoughFilePath = 'res/RG/EN-RG-65.txt'

        with open(rubensteinGoodenoughFilePath) as file:
            lines = file.readlines()

        wordPairs = []
        targetScores = []
        for line in lines:
            word0, word1, targetScore = tuple(line.strip().split('\t'))
            targetScore = float(targetScore)

            rubensteinGoodenoughData.append((word0, word1, targetScore))

    scores = []
    targetScores = []
    for word0, word1, targetScore in rubensteinGoodenoughData:
        if word0 in wordIndexMap and word1 in wordIndexMap:
            targetScores.append(targetScore)

            word0Index = wordIndexMap[word0]
            word1Index = wordIndexMap[word1]
            word0Embedding = embeddings[word0Index]
            word1Embedding = embeddings[word1Index]

            score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
            scores.append(score)

    if len(scores) == 0:
        return numpy.nan

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    rubensteinGoodenoughMetric = numpy.mean([pearson, spearman])

    return rubensteinGoodenoughMetric


wordSimilarity353Data = None
def wordSimilarity353(wordIndexMap, embeddings):
    global wordSimilarity353Data

    if wordSimilarity353Data is None:
        wordSimilarity353Data = []

        wordSimilarity353FilePath = 'res/WordSimilarity-353/combined.csv'
        data = pandas.read_csv(wordSimilarity353FilePath)

        for word0, word1, score in zip(data['Word1'], data['Word2'], data['Score']):
            wordSimilarity353Data.append((word0, word1, score))

    scores = []
    targetScores = []
    for word0, word1, targetScore in wordSimilarity353Data:
        if word0 in wordIndexMap and word1 in wordIndexMap:
            targetScores.append(targetScore)

            word0Index = wordIndexMap[word0]
            word1Index = wordIndexMap[word1]
            word0Embedding = embeddings[word0Index]
            word1Embedding = embeddings[word1Index]

            score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
            scores.append(score)

    if len(scores) == 0:
        return numpy.nan

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    metric = numpy.mean([pearson, spearman])

    return metric


simLex999Data = None
def simLex999(wordIndexMap, embeddings):
    global simLex999Data

    if simLex999Data is None:
        simLex999Data = []
        simLex999FilePath = 'res/SimLex-999/SimLex-999.txt'
        data = pandas.read_csv(simLex999FilePath, sep='\t')

        for word0, word1, targetScore in zip(data['word1'], data['word2'], data['SimLex999']):
            simLex999Data.append((word0, word1, targetScore))

    targetScores = []
    scores = []
    for word0, word1, targetScore in simLex999Data:
        if word0 in wordIndexMap and word1 in wordIndexMap:
            targetScores.append(targetScore)

            word0Index = wordIndexMap[word0]
            word1Index = wordIndexMap[word1]
            word0Embedding = embeddings[word0Index]
            word1Embedding = embeddings[word1Index]

            score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
            scores.append(score)

    if len(scores) == 0:
        return numpy.nan

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    simLex999Metric = numpy.mean([pearson, spearman])

    return simLex999Metric


syntacticWordData = None
def syntacticWordRelations(wordIndexMap, embeddings, maxWords=10):
    global syntacticWordData

    if syntacticWordData is None:
        syntacticWordData = []
        syntWordRelFilePath = 'res/Syntactic-Word-Relations/questions-words.txt'

        with open(syntWordRelFilePath, 'r') as file:
            lines = file.readlines()
            syntacticWordData = [tuple(line.lower().split(' ')) for line in lines if not line.startswith(':')]
            syntacticWordData = [(word0.strip(), word1.strip(), word2.strip(), word3.strip()) for word0, word1, word2, word3 in syntacticWordData]

    scores = []
    for word0, word1, word2, word3 in syntacticWordData:
        if word0 not in wordIndexMap or word1 not in wordIndexMap or word2 not in wordIndexMap or word3 not in wordIndexMap:
            continue

        word0Index = wordIndexMap[word0]
        word1Index = wordIndexMap[word1]
        word2Index = wordIndexMap[word2]
        word3Index = wordIndexMap[word3]

        word0Embedding = embeddings[word0Index]
        word1Embedding = embeddings[word1Index]
        word2Embedding = embeddings[word2Index]
        word3Embedding = embeddings[word3Index]

        similarity01 = vectors.cosineSimilarity(word0Embedding, word1Embedding)
        similarity23 = vectors.cosineSimilarity(word2Embedding, word3Embedding)

        score = 1
        minSimilarityDelta = abs(similarity01 - similarity23)
        for embedding in embeddings[:maxWords]:
            similarity2N = vectors.cosineSimilarity(word2Embedding, embedding)
            similarityDelta = abs(similarity01 - similarity2N)

            score = not (similarityDelta < minSimilarityDelta)
            if not score:
                break

        scores.append(score)

    if len(scores) == 0:
        return numpy.nan

    syntacticWordRelationsMetric = float(sum(scores)) / len(scores)

    return syntacticWordRelationsMetric


satQuestionsData = None
def satQuestions(wordIndexMap, embeddings):
    global satQuestionsData

    if satQuestionsData is None:
        satQuestionsData = []
        satQuestionsFilePath = 'res/SAT-Questions/SAT-package-V3.txt'

        maxLineLength = 50
        aCode = ord('a')

        with open(satQuestionsFilePath) as file:
            line = file.readline()
            while line != '':
                if len(line) < maxLineLength:
                    match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)
                    if match:
                        stemWord0, stemWord1 = match.group('word0'), match.group('word1')
                        satQuestion = [stemWord0, stemWord1]

                        line = file.readline()
                        match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)
                        while match:
                            choiceWord0, choiceWord1 = match.group('word0'), match.group('word1')
                            satQuestion.append(choiceWord0)
                            satQuestion.append(choiceWord1)

                            line = file.readline()
                            match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)

                        correctChoiceIndex = ord(line.strip()) - aCode
                        satQuestion.append(correctChoiceIndex)

                        satQuestionsData.append(satQuestion)

                line = file.readline()

    scores = []
    for satQuestion in satQuestionsData:
        if any([word not in wordIndexMap for word in satQuestion[:-1]]):
            continue

        stemWord0, stemWord1 = satQuestion[:2]

        stemWord0Index = wordIndexMap[stemWord0]
        stemWord1Index = wordIndexMap[stemWord1]
        stemWord0Embedding, stemWord1Embedding = embeddings[stemWord0Index], embeddings[stemWord1Index]
        stemSimilarity = vectors.cosineSimilarity(stemWord0Embedding, stemWord1Embedding)

        correctChoiceIndex = satQuestion[-1]
        choiceSimilarityDeltas = []

        choices = satQuestion[2:-1]
        for i in xrange(0, len(choices), 2):
            choiceWord0, choiceWord1 = choices[i], choices[i+1]
            choiceWord0Index, choiceWord1Index = wordIndexMap[choiceWord0], wordIndexMap[choiceWord1]
            choiceWord0Embedding, choiceWord1Embedding = embeddings[choiceWord0Index], embeddings[choiceWord1Index]

            choiceSimilarity = vectors.cosineSimilarity(choiceWord0Embedding, choiceWord1Embedding)

            choiceSimilarityDelta = abs(stemSimilarity - choiceSimilarity)
            choiceSimilarityDeltas.append(choiceSimilarityDelta)

            choiceIndex = numpy.argmin(choiceSimilarityDeltas)
            scores.append(int(choiceIndex == correctChoiceIndex))

    if len(scores) == 0:
        return numpy.nan

    metric = float(sum(scores)) / len(scores)

    return metric


def validate(wordIndexMap, embeddings):
    rg = rubensteinGoodenough(wordIndexMap, embeddings)
    sim353 = wordSimilarity353(wordIndexMap, embeddings)
    sl999 = simLex999(wordIndexMap, embeddings)
    syntRel = syntacticWordRelations(wordIndexMap, embeddings)
    sat = satQuestions(wordIndexMap, embeddings)

    return rg, sim353, sl999, syntRel, sat


def dump(metricsPath, epoch, superBatchIndex, *metrics, **customMetrics):
    rg, sim353, sl999, syntRel, sat = metrics
    metrics = [metric for metric in metrics if metric != metric]

    median = numpy.median(metrics)
    mean = numpy.mean(metrics)
    total = numpy.sum(metrics)

    metrics = {
        'epoch': epoch,
        'superBatchIndex': superBatchIndex,
        'rg': rg,
        'sim353': sim353,
        'sl999': sl999,
        'syntRel': syntRel,
        'sat': sat,
        'median': median,
        'mean': mean,
        'total': total
    }

    for name, value in customMetrics.items():
        metrics[name] = value

    metrics = [metrics]

    if os.path.exists(metricsPath):
        with open(metricsPath, 'a') as metricsFile:
            metricsHistory = pandas.DataFrame.from_dict(metrics)
            metricsHistory.to_csv(metricsFile, header=False)
    else:
        metricsHistory = pandas.DataFrame.from_dict(metrics)
        metricsHistory.to_csv(metricsPath, header=True)


def compareMetrics(metricsHistoryPath, *metricNames):
    metrics = pandas.DataFrame.from_csv(metricsHistoryPath)
    iterations = range(0, len(metrics))

    plt.grid()

    metricScatters = []
    colorNames = colors.cnames.keys()
    for metricIndex, metricName in enumerate(metricNames):
        metric = metrics[metricName]

        random.shuffle(colorNames)
        metricScatter = plt.scatter(iterations, metric, c=colorNames[metricIndex % len(colorNames)])
        metricScatters.append(metricScatter)

    metricsFileName = os.path.basename(metricsHistoryPath)
    plt.title(metricsFileName)

    plt.legend(metricScatters, metricNames, scatterpoints=1, loc='lower right', ncol=3, fontsize=8)

    plt.show()


def compareHistories(metricName, *metricsHistoryPaths):
    plt.grid()

    metricScatters = []
    metricsHistoryNames = []
    colorNames = colors.cnames.keys()

    for metricsHistoryIndex, metricsHistoryPath in enumerate(metricsHistoryPaths):
        metrics = pandas.DataFrame.from_csv(metricsHistoryPath)
        iterations = range(0, len(metrics))
        metric = metrics[metricName]

        random.shuffle(colorNames)
        metricScatter = plt.scatter(iterations, metric, c=colorNames[metricsHistoryIndex % len(colorNames)])
        metricScatters.append(metricScatter)

        metricsHistoryName = os.path.basename(metricsHistoryPath)
        metricsHistoryNames.append(metricsHistoryName)

    plt.title(metricName)
    plt.legend(metricScatters, metricsHistoryNames, scatterpoints=1, loc='lower right', ncol=3, fontsize=8)

    plt.show()


def main():
    embeddingsFilePath = '../data/Text8-vectors/vectors.bin'
    wordIndexMap, embeddings = kit.loadWord2VecEmbeddings(embeddingsFilePath)

    rgMetric = rubensteinGoodenough(wordIndexMap, embeddings)
    log.info('Rubenstein-Goodenough: {0:.2f}/10. State of the art: 9.15/10', rgMetric * 10)

    wordSim353Metric = wordSimilarity353(wordIndexMap, embeddings)
    log.info('WordSimilarity-353: {0:.2f}/10. State of the art: 8.1/10', wordSim353Metric * 10)

    simLex999Metric = simLex999(wordIndexMap, embeddings)
    log.info('SimLex-999: {0:.2f}/10. State of the art: 6.42/10', simLex999Metric * 10)

    syntWordRelMetric = syntacticWordRelations(wordIndexMap, embeddings)
    log.info('Syntactic word relations: {0:.2f}/10. State of the art: 10/10', syntWordRelMetric * 10)

    satQuestionsMetric = satQuestions(wordIndexMap, embeddings)
    log.info('SAT Questions: {0:.2f}/10. State of the art: 8.15/10', satQuestionsMetric * 10)


if __name__ == '__main__':
    main()