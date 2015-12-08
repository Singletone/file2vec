import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.io.{Path, File}

object main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("Processing")
    val spark = new SparkContext(conf)

    val extracted = s"../data/Experiments/Wikipedia-min/Extracted"

    val result = spark
      .wholeTextFiles("../data/Datasets/Wikipedia-min/*-wiki-*.txt.gz")
      .flatMap { case (dumpName, content) => Split(content) }
      .filter { case (title, text) => Filter(title, text) }
      .foreach { case (title, text) => File(textFile(extracted, title)).writeAll(text) }

    println(result)
  }

  def textFile(dirPath: String, fileName: String): Path = {
    dirPath + "/" + "[^\\w\\d\\(\\)]".r.replaceAllIn(fileName, "_")  + ".txt"
  }
}