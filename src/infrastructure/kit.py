from os.path import join


class PathTo:
    def __init__(self, dataDir, name):
        self.dataDir = dataDir
        self.name = name
        self.dataSetDir = join(self.dataDir, 'Datasets', name)

        self.injestedDir = join(self.dataDir, 'Injested')
        self.concatenatedDir = join(self.dataDir, 'Concatenated')
        self.processedDir = join(self.dataDir, 'Processed')
        self.parametersDir = join(self.dataDir, 'Parameters')
        self.metricsDir = join(self.dataDir, 'Metrics')

        self.concatenated = join(self.concatenatedDir, 'concatenated.txt')
        self.contexts = join(self.processedDir, 'contexts.bin')
        self.fileIndexMap = join(self.parametersDir, 'file_index_map.bin')
        self.fileEmbeddings = join(self.parametersDir, 'file_embeddings.bin')
        self.wordIndexMap = join(self.parametersDir, 'word_index_map.bin')
        self.wordEmbeddings = join(self.parametersDir, 'word_embeddings.bin')
        self.weights = join(self.parametersDir, 'weights.bin')


    def w2vEmbeddings(self, fileName):
        return join(self.dataDir, 'WordEmbeddings', fileName)


    def metrics(self, fileName):
        return join(self.metricsDir, fileName)