object Filter {
  def apply(title: String, text: String): Boolean = {
    val lowerTitle = title.toLowerCase
    val lowerText = text.toLowerCase

    val mayReferTo = s"$lowerTitle may refer to"
    val redirect = "#redirect"

    !(lowerTitle.contains(":") ||
      lowerText.startsWith(mayReferTo) ||
      lowerText.startsWith(redirect))
  }
}
