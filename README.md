All processing jobs can be shared between following workers:

1. Text extractor.
2. Metadata extractor.
3. Contexts reader.

Two first workers have their own processing task queues and work in parallel. Context reader may be launched only after first two workers have finished since it uses word metadata as one of it’s parameters.

##Text extractor
Input: data in original dataset format (bunch of zipped files, psts, database dump etc.).
Output: bunch of plain text files stored on disk; processing batches that are sumbitted to metadata extraction queue, processing batches that are sumbitted to context reading queue.

Depending on dataset, text extractor uses corresponding data connector to iterate through all text entries (which are usually represented as files). Once extracted text entry is cleaned from different garbage and is saved to a separate text file on disk. After saved, a metadata extraction task is submitted to metadata extraction queue.

To speed up data processing during text extraction phase, some minor text transformations may be applied that remove obviously corrupted parts of text. No complex text transformations should be done here.

Extractor packs text entries into batches and submits to the following worker.

Normally text extractor will unpack original dataset files (which may take 10Gb-100Gb) using data connector and create plain text file for each text entry (there may be millions of files).

**Pseudocode**:
```
for key, text in datasetConnector.next():
    key, text = cleanText(key, text)
    textFileName = saveToDisk(key, text)
    metadataExtractionTask = Task(key, textFileName)
    metadataExtractionQueue.submit(metadataExtractionTask)
    contextReadingTask = Task(key, textFileName)
    contextReadingQueue.submit(contextReadingTask)
```

##Metadata extractor
Input: batches of names/paths to plain text files.
Output: word vocabulary with word frequencies saved to disk.

Listens metadata extraction queue. For each incoming metadata extraction task metadata extractor reads corresponding text file stored on disk and extracts word frequencies alonf with other stats and informations. Extracted metadata should then be saved to disk.

**Pseudocode**:
```
wordVocabulary = {}
for mdExtractionBatch in metadataExtractionQueue:
    wordFrequencyMaps = spark.map(extractMD, mdExtractionBatch)
    wordVocabulary = spark.reduce(wordVocabulary, mergeMetadata)

saveToDisk(wordVocabulary)
```

##Contexts reader
Input: batches of names/paths to plain text files, word vocabulary with word frequencies.
Output: tensor with collection of contexts for each text file in a dataset.

Uses tokenizers and sliding windows that may implement different subsampling, pruning, tokanization and word context creation strategies. Depending on configuration, context reader may force each file context collection to have the same amount of contexts (by repeating existing contexts). All contexts are written to a tensor which is stored on disk.

**Pseudocode**:
```
wordVocabulary = loadFromDisk()

contextTensor = HDF5.createTensor(
    FILES_COUNT, 
    CONTEXTS_COUNT, 
    WINDOW_SIZE, 
    CONTEXTS_TENSOR_PATH)

for key, text in contextReadingQueue.next():
    tokenizer = StanfordNLPTokenizer()
    slidingWindow = SlidingWindow(text, tokenizer, WINDOW_SIZE)
    for context in slidingWindow.next(text):
        contextTensor[textIndex, contextIndeex] = context
```

##Technologies
Processing queue persistence
So far the best option here is Celery library. It has everything needed to build distributed async processing queues with a bunch of monitoring and troubleshooting tools.

###Parallel processing
This is where Spark can be useful. Each worker may submit processing batches and each of these batches may be processed in parallel using Spark.

###Large tensor storage
The best choise here is probably HDF5. This is a logical file system that provides seamless interaction with multidimansinal data stored on disk. Interface that is identical to Numpy is a huge plus.

###Tokenization
This an important processing stage. As the most simple option text may be splitted into words using spaces. But there are more promising approaches. One of them is to tokenize text with semantics in mind. This will allow to preserve concepts and different named entities which may increase model quality. One of the best tool that may be used for this is a Stanford NLP.