import os
import glob
import gzip
import re


class WikipediaSourceConnector:
    def __init__(self, inputDirectoryPath):
        pathName = inputDirectoryPath + '/*.txt.gz'
        dumpPaths = glob.glob(pathName)

        self.dumpPaths = glob.glob(pathName)
        self.total = len(dumpPaths)


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
            pages = filter(WikipediaSourceConnector.filterPage, pages)
        except:
            pass

        return dumpName, pages


    @staticmethod
    def cleanText(name, text):
        name = re.sub('[^_a-zA-Z0-9\s\(\)]', '', name).strip()

        restrictedHeaders = ['see also', 'footnotes', 'references', 'further reading', 'external links', 'books']

        headings = [name] + re.findall('^=+\s*([^=]+)\s*=+$', text, flags=re.M)
        paragraphs = re.split('^=+\s*[^=]+\s*=+$', text, flags=re.M)

        text = ''

        for heading, paragraph in zip(headings, paragraphs):
            if heading.lower() not in restrictedHeaders:
                text += paragraph.lower()

        text = re.sub('\s+', ' ', text)
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
        text = re.sub('\([^\)]+\)', '', text)
        text = re.sub('(:[^\.]\.)', '', text)
        text = re.sub('[,":_\*]', ' ', text)
        text = re.sub('!', '.', text)
        text = re.sub('\?', '.', text)
        text = re.sub('\s(\.{4,})\s', ' ', text)
        text = re.sub('\s(\.{3})\s', '.', text)
        text = re.sub('\s(\.{2})\s', ' ', text)
        text = re.sub('<[^>]+>', '', text)
        text = re.sub('([0-9\-]+)s', ' NUMBER ', text)
        text = re.sub('([0-9\-]+)th', ' NUMBER ', text)
        text = re.sub('[^a-z]+([0-9\-]+)[^a-z]+', ' NUMBER ', text)
        text = re.sub('\s([^a-zA-Z0-9\.\-\s]+)\s', ' SYMBOL ', text)
        text = re.sub('\s([bcdefghjklmnopqrstuvwxyz])\s', ' SYMBOL ', text)

        sentences = re.split('[(\n{2,})\.;]', text)
        sentences = [re.sub('[\s]+', ' ', sentence).strip() for sentence in sentences]
        sentences = [sentence for sentence in sentences
                     if len(sentence.split(' ')) > 5 and sentence.count('NUMBER') < 3]

        text = '. '.join(sentences) + '.'

        return name, text


    def next(self, clean=True):
        for dumpIndex, dumpPath in enumerate(self.dumpPaths):
            dumpName, pages = WikipediaSourceConnector.unpackDump(dumpPath)

            if any(pages):
                for name, text in pages:
                    if clean:
                        name, text = WikipediaSourceConnector.cleanText(name, text)

                    yield name, text