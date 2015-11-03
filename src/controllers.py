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
    pathTo = kit.PathTo('Cockatoo', 'wiki_full_s800_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(
        connector = connectors.TextFilesConnector(pathTo.dataSetDir),
        sample=1e1,
        minCount=1,
        windowSize=5,
        negative=100,
        fileEmbeddingSize=10000,
        epochs=50,
        batchSize=1,
        learningRate=0.025
    )

    controller = DataPreparationController()
    controller.launch(pathTo, hyper)