import kit
import connectors
import parameters
import extraction
import weeding
import processing


pathTo = kit.PathTo('RottenTomatos-Kaggle', 'rtk')
connector = connectors.RottenTomatosConnector(pathTo.dataSetDir)

print connector.count()