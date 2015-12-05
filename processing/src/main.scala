import java.io.StringReader

import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation
import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.process.{PTBTokenizer, CoreLabelTokenFactory}
import org.apache.spark.{SparkConf, SparkContext}

object main {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .setAppName("Wikipedia ETL")
      .setMaster("local[4]")

    val spark = new SparkContext(sparkConf)

    val path = "../data/Datasets/Wikipedia-min/*-wiki-*.txt.gz"

    val words = spark
      .wholeTextFiles(path)
      .flatMap({ case (fileName, content) => Split(content) })
      .flatMap({ case (title, text) => Tokenize(text) })
      .map(word => (word.toLowerCase, 1))
      .reduceByKey(_ + _)
      .sortBy({ case (word, count) => count }, ascending=false)
      .collect()

    words.foreach({ case (word, count) => println((word, count)) })
  }
}