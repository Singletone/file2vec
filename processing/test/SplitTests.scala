import org.scalatest.FlatSpec

class SplitTests extends FlatSpec {

  "Splitter" should "split empty dump into an empty array" in {
    assert(Split("").length == 0)
  }

  it should "return correct number of text entries" in {
    assert(Split("[[T-90]]").length == 1)
    assert(Split("[[Merkava]][[SU-27]]").length == 2)
    assert(Split("[[MiG-29]]About MiG.[[Cockatoo]]About cockatoo.").length == 2)
  }

  it should "create separate titles and texts" in {
    assert(Split("[[T-90]]About T-90.").sameElements(Array(("T-90", "About T-90."))))
    assert(Split("[[T-90]]About T-90.[[MiG-29]]About MiG.").sameElements(
      Array(("T-90", "About T-90."), ("MiG-29", "About MiG."))))
    assert(Split("[[T-90]]About T-90.[[MiG-29]]About MiG.[[Empty]]").sameElements(
      Array(("T-90", "About T-90."), ("MiG-29", "About MiG."), ("Empty", ""))))
  }

  it should "split title tag and text with title extracted" in {
    assert(Split.asTitleText("[[Empty]]") == ("Empty", ""))
    assert(Split.asTitleText("[[T-90]]About T-90.") == ("T-90", "About T-90."))
  }

  it should "extract title" in {
    assert(Split.asTitle("[[Empty]]") == "Empty")
  }
}