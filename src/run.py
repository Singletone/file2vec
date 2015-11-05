from kit import *
from parameters import *
from validation import *
from vectors import *


if __name__ == '__main__':
    pathTo = PathTo('Cockatoo', experiment='default', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    fileIndexMap = loadWordMap(pathTo.fileIndexMap)
    fileEmbeddings = loadEmbeddings(pathTo.fileEmbeddings)

    comparator = lambda a, b: cosineSimilarity(a, b)

    compareEmbeddings(fileIndexMap, fileEmbeddings, comparator=comparator, annotate=False, axisLabels=True)
    # compareMetrics(pathTo.metrics('history.csv'), 'meanError', 'medianError', 'minError', 'maxError')
