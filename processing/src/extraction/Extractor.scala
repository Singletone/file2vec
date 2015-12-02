package extraction

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

abstract class Extractor(spark: SparkContext) {
  def extract(path: String): RDD[String]
}