from multiprocessing import Process

import kit
import extraction
import weeding
import processing
import traininig


class DataPreparationController:
    def launch(self, pathTo):
        targets = [extraction.launch, weeding.launch, processing.launch]

        for target in targets:
            proces = Process(target=target, args=(pathTo,))
            proces.start()
            proces.join()

        traininig.launch(pathTo)


if __name__ == '__main__':
    pathTo = kit.PathTo('Cockatoo')

    controller = DataPreparationController()
    controller.launch(pathTo)