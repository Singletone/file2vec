from infrastructure import kit
from injestion.connectors import *


def launch(connector, concatenatedFilePath, injestedDirPath):
    pass

if __name__ == '__main__':
    pathTo = kit.PathTo('../data', 'Duplicates')

    connector = TextFilesConnector(pathTo.dataSetDir)

    launch(connector, pathTo.concatenated, pathTo.injestedDir)