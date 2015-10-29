import os
from os.path import join


class PathTo:
    def __init__(self, name):
        self.name = name
        self.dataDir = '../data'
        self.baseDir = join(self.dataDir, name)

        self.rawDir = join(self.baseDir, 'Raw')
        self.preparedDir = join(self.baseDir, 'Prepared')
        self.concatenatedDir = join(self.baseDir, 'Concatenated')
        self.processedDir = join(self.baseDir, 'Processed')
        self.parametersDir = join(self.baseDir, 'Parameters')
        self.metricsDir = join(self.baseDir, 'Metrics')

        self.ensureDirectories(
            self.rawDir,
            self.preparedDir,
            self.concatenatedDir,
            self.processedDir,
            self.parametersDir,
            self.metricsDir)

        self.concatenated = join(self.concatenatedDir, 'concatenated.txt')
        self.contexts = join(self.processedDir, 'contexts.bin')
        self.fileIndexMap = join(self.parametersDir, 'file_index_map.bin')
        self.fileEmbeddings = join(self.parametersDir, 'file_embeddings.bin')
        self.wordIndexMap = join(self.parametersDir, 'word_index_map.bin')
        self.wordEmbeddings = join(self.parametersDir, 'word_embeddings.bin')
        self.weights = join(self.parametersDir, 'wights.bin')


    @staticmethod
    def ensureDirectories(*directories):
        for directory in directories:
            if not os.path.exists(directory):
                os.mkdir(directory)
                os.chown(directory, 1000, 1000)


    def w2vEmbeddings(self, fileName):
        return join(self.dataDir, 'WordEmbeddings', fileName)


    def metrics(self, fileName):
        return join(self.metricsDir, fileName)