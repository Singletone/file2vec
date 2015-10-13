import os
import kit
import parameters


def convertWord2Vec(word2VecEmbeddingsFilePath, vocabularyFilePath, embeddingsFilePath):
    if os.path.exists(vocabularyFilePath):
        os.remove(vocabularyFilePath)

    if os.path.exists(embeddingsFilePath):
        os.remove(embeddingsFilePath)

    vocabulary, embeddings = kit.loadWord2VecEmbeddings(word2VecEmbeddingsFilePath)

    parameters.dumpWordVocabulary(vocabulary, vocabularyFilePath, dumpFrequency=False)
    parameters.dumpEmbeddings(embeddings, embeddingsFilePath)


if __name__ == '__main__':
    word2VecEmbeddingsFilePath = '../data/Drosophila/Parameters/drosophila_w2v.bin'
    vocabularyFilePath = '../data/Drosophila/Parameters/word_vocabulary.bin'
    embeddingsFilePath = '../data/Drosophila/Parameters/word_embeddings.bin'

    convertWord2Vec(word2VecEmbeddingsFilePath, vocabularyFilePath, embeddingsFilePath)