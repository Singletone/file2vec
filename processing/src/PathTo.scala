import java.nio.file.Paths

class PathTo(dataDir: String, datasetName: String, experimentName: String) {
  def textFile(dirPath: String, title: String): String = {
    Paths.get(dataDir, datasetName + ".txt").toString
  }

  def dataset = Paths.get(dataDir, datasetName).toString
  def experiments = Paths.get(dataDir, "Experiments").toString
  def experiment = Paths.get(experiments, experimentName).toString
  def extracted = Paths.get(experiment, "Extracted").toString
}
