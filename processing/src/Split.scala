object Split {
  val articlePattern = "(\\[\\[[^\\]]+\\]\\])([^\\[]*)".r
  val bracketsPattern = "[\\[\\]]".r

  def apply(dump: String): Array[(String, String)] = {
    articlePattern
      .findAllIn(dump)
      .map(asTitleText)
      .toArray
  }

  def asTitleText(article: String): (String, String) = {
    val articlePattern(title, text) = article

    (asTitle(title), text)
  }

  def asTitle(title: String): String = {
    bracketsPattern.replaceAllIn(title, "")
  }
}
