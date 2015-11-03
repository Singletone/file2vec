import os
import glob
import gzip
import re
import numpy


class TextFilesConnector:
    def __init__(self, inputDirectoryPath):
        pathName = inputDirectoryPath + '/*.txt'
        self.textFilePaths = glob.glob(pathName)


    def count(self):
        return len(self.textFilePaths)


    def iterate(self):
        for textFileIndex, textFilePath in enumerate(self.textFilePaths):
            with open(textFilePath, 'r') as textFile:
                textFileName = os.path.basename(textFilePath).split('.')[0]
                text = textFile.read()

                yield textFileIndex, textFileName, text


class WikipediaConnector:
    def __init__(self, inputDirectoryPath):
        pathName = inputDirectoryPath + '/*.txt.gz'
        self.dumpPaths = glob.glob(pathName)

        numpy.random.seed(123)
        numpy.random.shuffle(self.dumpPaths)

        self.dumpPaths = self.dumpPaths[:2]


    @staticmethod
    def filterPage(page):
        name, text = page

        if ':' in name:
            return False

        mayReferTo = '{0} may refer to'.format(name).lower()
        if text.startswith(mayReferTo):
            return False

        if text.startswith('#redirect'):
            return False

        if len(text) < 10:
            return False

        return True


    @staticmethod
    def unpackDump(dumpPath):
        dumpName = os.path.basename(dumpPath).split('.')[0]
        pages = []

        try:
            with gzip.open(dumpPath, 'rb') as dumpFile:
                dumpText = dumpFile.read()

            names = [name.strip() for name in re.findall('^\[\[(?P<title>[^\]]+)\]\]\s?$', dumpText, flags=re.M)]

            texts = [text.strip() for text in re.split('^\[\[[^\]]+\]\]\s?$', dumpText, flags=re.M) if text]

            pages = zip(names, texts)
            pages = filter(WikipediaConnector.filterPage, pages)
        except:
            pass

        return dumpName, pages


    @staticmethod
    def stripWikiMarkup(name, text):
        name = re.sub('[^_a-zA-Z0-9\s\(\)]', '', name).strip()

        restrictedHeaders = ['see also', 'footnotes', 'references', 'further reading', 'external links', 'books']

        headings = [name] + re.findall('^=+\s*([^=]+)\s*=+$', text, flags=re.M)
        paragraphs = re.split('^=+\s*[^=]+\s*=+$', text, flags=re.M)

        text = ''

        for heading, paragraph in zip(headings, paragraphs):
            if heading.lower() not in restrictedHeaders:
                text += paragraph

        return name, text


    def count(self):
        return len(self.dumpPaths)


    def iterate(self):
        for dumpIndex, dumpPath in enumerate(self.dumpPaths):
            dumpName, pages = WikipediaConnector.unpackDump(dumpPath)

            if any(pages):
                for name, text in pages:
                    name, text = WikipediaConnector.stripWikiMarkup(name, text)

                    yield dumpIndex, name, text