package extraction

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

abstract class Extractor(spark: SparkContext) {
  val tokenizer = new SimpleTokenizer()

  def extract(path: String): RDD[(String, String)]
}