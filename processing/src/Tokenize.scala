import java.io.StringReader

import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.process.{CoreLabelTokenFactory, PTBTokenizer}

object Tokenize {
  def apply(text: String): Array[String] = {
    val stringReader = new StringReader(text)
    val tokenFactory = new CoreLabelTokenFactory()

    val tokenizer = new PTBTokenizer[CoreLabel](stringReader, tokenFactory, "")

    tokenizer
      .tokenize()
      .toArray()
      .map(token => token.toString)
  }
}