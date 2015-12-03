package extraction

abstract class SlidingWindow(tokenizer: Tokenizer) {
  def sliding(text: String, size: Int): Array[Array[String]] = {
    sentences(text)
      .flatMap(sentence => tokenizer.tokenize(sentence))
      .map(tokens => tokens.sliding(size).toArray)
  }

  def sentences(text: String): Array[String]
}
