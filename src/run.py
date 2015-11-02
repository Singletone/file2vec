from kit import *
from parameters import *
from validation import *
from vectors import *


if __name__ == '__main__':
    pathTo = PathTo('Duplicates')
    fileIndexMap = loadIndexMap(pathTo.fileIndexMap)
    fileEmbeddings = loadEmbeddings(pathTo.fileEmbeddings)

    compareEmbeddings(fileIndexMap, fileEmbeddings, annotate=True)
    # compareMetrics(pathTo.metrics('history.csv'), 'error')