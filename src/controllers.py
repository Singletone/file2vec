from multiprocessing import Process

import kit
import parameters
import connectors
import extraction
import weeding
import processing
import traininig


class DataPreparationController:
    def launch(self, pathTo, hyper):
        targets = [extraction.launch, weeding.launch, processing.launch]

        for target in targets:
            proces = Process(target=target, args=(pathTo, hyper))
            proces.start()
            proces.join()

        traininig.launch(pathTo, hyper)


if __name__ == '__main__':
    pathTo = kit.PathTo('Wikipedia', experiment='wiki_5_dumps', w2vEmbeddings='wiki_full_s800_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(
        connector = connectors.WikipediaConnector(pathTo.dataSetDir),
        sample=1e1,
        minCount=1,
        windowSize=3,
        negative=100,
        strict=False,
        fileEmbeddingSize=10000,
        epochs=5,
        batchSize=1,
        learningRate=0.025
    )

    controller = DataPreparationController()
    controller.launch(pathTo, hyper)