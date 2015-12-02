import extraction.WikipediaDumpSplitter
import org.scalatest.FlatSpec

class WikipediaDumpSplitterTests extends FlatSpec {

  "Dump splitter" should "split empty dump into an empty array" in {
    assert(WikipediaDumpSplitter("").length == 0)
  }

  it should "return correct number of articles" in {
    assert(WikipediaDumpSplitter("[[T-90]]").length == 1)
    assert(WikipediaDumpSplitter("[[Merkava]][[SU-27]]").length == 2)
    assert(WikipediaDumpSplitter("[[MiG-29]]About MiG.[[Cockatoo]]About cockatoo.").length == 2)
  }

  it should "create separate titles and bodies" in {
    assert(WikipediaDumpSplitter("[[T-90]]About T-90.").sameElements(Array(("T-90", "About T-90."))))
    assert(WikipediaDumpSplitter("[[T-90]]About T-90.[[MiG-29]]About MiG.").sameElements(
      Array(("T-90", "About T-90."), ("MiG-29", "About MiG."))))
    assert(WikipediaDumpSplitter("[[T-90]]About T-90.[[MiG-29]]About MiG.[[Empty]]").sameElements(
      Array(("T-90", "About T-90."), ("MiG-29", "About MiG."), ("Empty", ""))))
  }

  it should "split title tag from body and extract title" in {
    assert(WikipediaDumpSplitter.asTitleBody("[[Empty]]") == ("Empty", ""))
    assert(WikipediaDumpSplitter.asTitleBody("[[T-90]]About T-90.") == ("T-90", "About T-90."))
  }

  it should "extract title" in {
    assert(WikipediaDumpSplitter.asTitle("[[Empty]]") == "Empty")
  }
}