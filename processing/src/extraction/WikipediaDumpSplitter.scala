package extraction

object WikipediaDumpSplitter {
  def apply(dump: String): Array[(String)] = {
    "\\[\\[[^\\]+]\\]\\]".r.findAllIn(dump).toArray
  }
}
