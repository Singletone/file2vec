package extraction

abstract class Tokenizer {
  def tokenize(text: String): Array[String]
}
