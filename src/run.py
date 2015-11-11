from binary import *
from parameters import *
from validation import *
from vectors import *


if __name__ == '__main__':
    pathTo = kit.PathTo('Cockatoo', experiment='cockatoo', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    # pathTo = kit.PathTo('Duplicates', experiment='duplicates', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    textIndexMap = loadMap(pathTo.textIndexMap)
    fileEmbeddings = loadTensor(pathTo.fileEmbeddings)

    comparator = lambda a, b: cosineSimilarity(a, b) + 1 / euclideanDistance(a, b)

    compareEmbeddings(textIndexMap, fileEmbeddings, comparator=comparator, annotate=False, axisLabels=True)
    # compareMetrics(pathTo.metrics('history.csv'), 'meanError', 'medianError', 'minError', 'maxError')
