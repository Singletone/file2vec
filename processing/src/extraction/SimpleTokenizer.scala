package extraction

class SimpleTokenizer extends Tokenizer {
  val wordSeparator = "\\s".r

  def tokenize(text: String): Array[String] = {
    wordSeparator.split(text)
  }
}