import os
from os.path import join


class PathTo:
    def __init__(self, datasetName, w2vEmbeddings=''):
        self.datasetName = datasetName
        self.w2vEmbeddings = w2vEmbeddings
        self.dataDir = '../data'
        self.dataSetDir = join(self.dataDir, 'Datasets', datasetName)

        self.extractedDir = join(self.dataDir, 'Extracted')
        self.weededDir = join(self.dataDir, 'Weeded')
        self.concatenatedDir = join(self.dataDir, 'Concatenated')
        self.processedDir = join(self.dataDir, 'Processed')
        self.parametersDir = join(self.dataDir, 'Parameters')
        self.metricsDir = join(self.dataDir, 'Metrics')
        self.w2vEmbeddingsDir = join(self.dataDir, 'WordEmbeddings')

        self.ensureDirectories(
            self.extractedDir,
            self.weededDir,
            self.concatenatedDir,
            self.processedDir,
            self.parametersDir,
            self.metricsDir,
            self.w2vEmbeddingsDir)

        self.concatenated = join(self.concatenatedDir, 'concatenated.txt')
        self.contexts = join(self.processedDir, 'contexts.bin')
        self.fileIndexMap = join(self.parametersDir, 'file_index_map.bin')
        self.fileEmbeddings = join(self.parametersDir, 'file_embeddings.bin')
        self.wordIndexMap = join(self.parametersDir, 'word_index_map.bin')
        self.wordEmbeddings = join(self.parametersDir, 'word_embeddings.bin')
        self.weights = join(self.parametersDir, 'weights.bin')
        self.w2vEmbeddings = join(self.w2vEmbeddingsDir, self.w2vEmbeddings)


    @staticmethod
    def ensureDirectories(*directories):
        for directory in directories:
            if not os.path.exists(directory):
                os.mkdir(directory)
                os.chown(directory, 1000, 1000)


    def metrics(self, fileName):
        return join(self.metricsDir, fileName)