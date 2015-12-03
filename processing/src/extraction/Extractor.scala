package extraction

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

abstract class Extractor(spark: SparkContext) {
  val tokenizer = new WhiteSpaceTokenizer()
  val slidingWindow = new DotSlidingWindow(tokenizer)

  def extract(path: String): RDD[(String, String)]
}