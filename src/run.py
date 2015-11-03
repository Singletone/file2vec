from kit import *
from parameters import *
from validation import *
from vectors import *


if __name__ == '__main__':
    pathTo = PathTo('Duplicates', 'wiki_full_s800_w10_mc20_hs1.bin')
    fileIndexMap = loadIndexMap(pathTo.fileIndexMap)
    fileEmbeddings = loadEmbeddings(pathTo.fileEmbeddings)

    compareEmbeddings(fileIndexMap, fileEmbeddings, annotate=True)
    # compareMetrics(pathTo.metrics('history.csv'), 'meanError', 'medianError', 'minError', 'maxError')
