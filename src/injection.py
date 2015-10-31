import os
import shutil
import time

import kit
import log
import connectors


def saveText(filePath, text):
    with open(filePath, 'a+') as pageFile:
        if pageFile.tell():
            pageFile.write(' ')

        pageFile.write(text)


def extractConnectors(outputDirectoryPath, outputConcatFilePath, *connectors):
    if os.path.exists(outputDirectoryPath):
        shutil.rmtree(outputDirectoryPath, ignore_errors=True)

    os.mkdir(outputDirectoryPath)
    os.chown(outputDirectoryPath, 1000, 1000)

    if os.path.exists(outputConcatFilePath):
        os.remove(outputConcatFilePath)

    totals = map(lambda connector: connector.total, connectors)
    total = sum(totals)

    log.info('Found {0} dumps.', total)

    startTime = time.time()

    progress = 0
    pagesCount = 0
    for connector in connectors:
        progress += 1

        for name, text in connector.next(clean=True):
            outputFilePath = os.path.join(outputDirectoryPath, name + '.txt')

            saveText(outputFilePath, text)
            saveText(outputConcatFilePath, text)

            currentTime = time.time()
            elapsed = currentTime - startTime
            pagesCount += 1

            log.progress('Extracting connectors: {0:.3f}%. Elapsed: {1}. Pages: {2}.',
                         progress,
                         total,
                         log.delta(elapsed),
                         pagesCount)

    log.lineBreak()


if __name__ == '__main__':
    pathTo = kit.PathTo('Cockatoo')

    wikiConnector = connectors.WikipediaSourceConnector(pathTo.dataSetDir)

    extractConnectors(pathTo.preparedDir, pathTo.concatenated, wikiConnector)