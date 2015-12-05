import org.scalatest.FlatSpec

class TokenizeTests extends FlatSpec {
  "Tokenizer" should "respect spaces and puntuation" in {
    assert(Tokenize("Time is an illusion. Lunchtime doubly so.").sameElements(
      Array("Time", "is", "an", "illusion", ".", "Lunchtime", "doubly", "so", ".")))
  }
}
