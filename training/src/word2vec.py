import io
import numpy as np
import progress


def load(filePath, loadEmbeddings=True):
    with open(filePath, 'rb') as w2vFile:
        firstLine = w2vFile.readline()
        embeddingsCount, embeddingSize = tuple(firstLine.split(' '))
        embeddingsCount, embeddingSize = int(embeddingsCount), int(embeddingSize)
        wordIndexMap = {}
        embeddings = np.zeros((embeddingsCount, embeddingSize))

        with progress.start('Loading W2V embeddings: %(percentage)i%%. %(value)i embeddings %(size)i features each.', embeddingsCount) as update:
            embeddingIndex = 0
            while True:
                word = ''
                while True:
                    char = w2vFile.read(1)

                    if not char:
                        if loadEmbeddings:
                            return wordIndexMap, embeddings
                        else:
                            return wordIndexMap

                    if char == ' ':
                        word = word.strip()
                        break

                    word += char

                wordIndexMap[word] = len(wordIndexMap)
                if loadEmbeddings:
                    embedding = np.fromfile(w2vFile, dtype='float32', count=embeddingSize)
                    embeddings[wordIndexMap[word]] = embedding
                else:
                    w2vFile.seek(embeddingSize * 4, io.SEEK_CUR)

                embeddingIndex += 1

                update(embeddingIndex, size=embeddingSize)