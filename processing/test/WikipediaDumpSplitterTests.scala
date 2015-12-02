import extraction.WikipediaDumpSplitter
import org.scalatest.FlatSpec

class WikipediaDumpSplitterTests extends FlatSpec {

  "Dump splitter" should "split empty dump into an empty array" in {
    assert(WikipediaDumpSplitter("").length == 0)
  }

  it should "return correct number of content" in {
    assert(WikipediaDumpSplitter("[[0]]").length == 1)
    assert(WikipediaDumpSplitter("[[0]][[1]]").length == 2)
    assert(WikipediaDumpSplitter("[[0]]0[[1]]1").length == 2)
  }
}