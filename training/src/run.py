import word2vec

filePath = '../../data/WordEmbeddings/wiki_full_s200_w10_mc20_hs1.bin'
wordIndexMap, embeddings = word2vec.load(filePath)