import scala.reflect.io.File

class WikipediaContext(pathTo: PathTo, hyperParams: HyperParameters)
  extends ProcessingContext(pathTo, hyperParams) {

  def extract() = {
    this.spark
      .wholeTextFiles(this.pathTo.dataset)
      .flatMap { case (fileName, content) => Split(content) }
      .map { case (title, text) => (this.pathTo.textFile(this.pathTo.extracted, title), text) }
      .foreach { case (filePath, content) => File(filePath).writeAll(content) }
  }
}