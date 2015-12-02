package extraction

import java.nio.file.Paths

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class WikipediaExtractor(spark: SparkContext) extends Extractor(spark) {
  def extract(path: String): RDD[String] = {
    val pattern = Paths.get(path, "*-wiki-*.txt.gz").toString

    spark
      .wholeTextFiles(pattern)
      .map { case (fileName, content) => content }
  }
}