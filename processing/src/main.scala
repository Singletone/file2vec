object main {
  def main(args: Array[String]) {
    val pathTo = new PathTo("../data", "Wikipedia-min", "Wikipedia-min")
    val hyperParams = new HyperParameters()

    val wikipediaContext = new WikipediaContext(pathTo, hyperParams)

    wikipediaContext.process()
  }
}