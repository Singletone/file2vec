import os
import shutil
import glob
import time
import re
import gzip

import log


def filterPage(page):
    pageName, pageText = page

    if ':' in pageName:
        return False

    mayReferTo = '{0} may refer to'.format(pageName).lower()
    if pageText.startswith(mayReferTo):
        return False

    if pageText.startswith('#redirect'):
        return False

    if len(pageText) < 10:
        return False

    return True


def cleanPage(page):
    pageName, pageText = page

    pageName = re.sub('[^_a-zA-Z0-9\s\(\)]', '', pageName).strip()

    restrictedHeaders = ['see also', 'footnotes', 'references', 'further reading', 'external links', 'books']

    headings = [pageName] + re.findall('^=+\s*([^=]+)\s*=+$', pageText, flags=re.M)
    paragraphs = re.split('^=+\s*[^=]+\s*=+$', pageText, flags=re.M)

    pageText = ''

    for heading, paragraph in zip(headings, paragraphs):
        if heading.lower() not in restrictedHeaders:
            pageText += paragraph.lower()

    pageText = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', pageText)
    pageText = re.sub('\([^\)]+\)', '', pageText)
    pageText = re.sub('(:[^\.]\.)', '', pageText)
    pageText = re.sub('[,":_\*]', ' ', pageText)
    pageText = re.sub('!', '.', pageText)
    pageText = re.sub('\?', '.', pageText)
    pageText = re.sub('\s(\.{4,})\s', ' ', pageText)
    pageText = re.sub('\s(\.{3})\s', '.', pageText)
    pageText = re.sub('\s(\.{2})\s', ' ', pageText)
    pageText = re.sub('<[^>]+>', '', pageText)
    pageText = re.sub('([0-9\-]+)s', ' NUMBER ', pageText)
    pageText = re.sub('[^a-z]+([0-9\-]+)[^a-z]+', ' NUMBER ', pageText)
    pageText = re.sub('\s([^a-zA-Z0-9\.\-\s]+)\s', ' SYMBOL ', pageText)
    pageText = re.sub('\s([bcdefghjklmnopqrstuvwxyz])\s', ' SYMBOL ', pageText)

    sentences = re.split('[(\n{2,})\.;]', pageText)
    sentences = [re.sub('[\s]+', ' ', sentence).strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences
                 if len(sentence.split(' ')) > 5 and sentence.count('NUMBER') < 3]

    pageText = '. '.join(sentences) + '.'

    return pageName, pageText


def savePage(dumpDirectoryPath, pageName, pageText):
    filePath = os.path.join(dumpDirectoryPath, pageName + '.txt')

    with open(filePath, 'a+') as file:
        if file.tell():
            file.write(' ')

        file.write(pageText)


def unpackDump(dumpPath, cleanText):
    dumpName = os.path.basename(dumpPath).split('.')[0]
    pages = []

    try:
        with gzip.open(dumpPath, 'rb') as dumpFile:
            dumpText = dumpFile.read()

        pageNames = [name.strip() for name in re.findall('^\[\[(?P<title>[^\]]+)\]\]\s?$', dumpText, flags=re.M)]

        pageTexts = [pageText.strip() for pageText in re.split('^\[\[[^\]]+\]\]\s?$', dumpText, flags=re.M) if pageText]

        pages = zip(pageNames, pageTexts)
        pages = filter(filterPage, pages)

        if cleanText:
            pages = map(cleanPage, pages)
    except:
        pass

    return dumpName, pages


def prepareWikipediaDumps(inputDirectoryPath, outputDirectoryPath, outputConcatFilePath, cleanText=True):
    if os.path.exists(outputDirectoryPath):
        shutil.rmtree(outputDirectoryPath, ignore_errors=True)
    os.mkdir(outputDirectoryPath)
    os.chown(outputDirectoryPath, 1000, 1000)

    if os.path.exists(outputConcatFilePath):
        os.remove(outputConcatFilePath)

    pathName = inputDirectoryPath + '/*wiki*.txt.gz'
    dumpPaths = glob.glob(pathName)
    dumpsCount = len(dumpPaths)
    pagesCount = 0

    log.info('Found {0} Wikipedia dumps.', dumpsCount)

    startTime = time.time()

    for dumpIndex, dumpPath in enumerate(dumpPaths):
        dumpName, pages = unpackDump(dumpPath, cleanText)

        if any(pages):
            pageDirectoryPath = os.path.join(outputDirectoryPath, dumpName)
            os.mkdir(pageDirectoryPath)
            os.chown(pageDirectoryPath, 1000, 1000)

            for pageName, pageText in pages:
                outputFilePath = os.path.join(pageDirectoryPath, pageName + '.txt')

                with open(outputFilePath, 'w+') as outputConcatFile:
                    if outputConcatFile.tell():
                        outputConcatFile.write(' ')

                    outputConcatFile.write(pageText)

                with open(outputConcatFilePath, 'a+') as outputConcatFile:
                    if outputConcatFile.tell():
                        outputConcatFile.write(' ')

                    outputConcatFile.write(pageText)

                pagesCount += 1

            currentTime = time.time()
            elapsed = currentTime - startTime
            secondsPerFile = elapsed / (dumpIndex + 1)

            log.progress('Unpacking Wikipedia dumps: {0:.3f}%. Elapsed: {1} ({2:.3f} sec/dump). Pages: {3}.',
                         dumpIndex + 1,
                         dumpsCount,
                         log.delta(elapsed),
                         secondsPerFile,
                         pagesCount)

    log.lineBreak()

if __name__ == '__main__':
    prepareWikipediaDumps(
        inputDirectoryPath = '../data/Drosophila/Raw',
        outputDirectoryPath = '../data/Drosophila/Prepared',
        outputConcatFilePath = '../data/Drosophila/Concatenated/drosophila.txt')