package extraction

object WikipediaDumpSplitter {
  val articlePattern = "(\\[\\[[^\\]]+\\]\\])([^\\[]*)".r
  val bracketsPattern = "[\\[\\]]".r

  def apply(dump: String): Array[(String, String)] = {
    articlePattern
      .findAllIn(dump)
      .map(asTitleBody)
      .toArray
  }

  def asTitleBody(article: String): (String, String) = {
    val articlePattern(title, body) = article

    (asTitle(title), body)
  }

  def asTitle(title: String): String = {
    bracketsPattern.replaceAllIn(title, "")
  }
}
