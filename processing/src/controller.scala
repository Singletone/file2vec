import org.apache.spark._

object controller {
  def main(args: Array[String]) {
    val path = "../data/Datasets/Wikipedia-min/*.txt.gz"

    val conf = new SparkConf().setAppName("Spark Pi").setMaster("local[4]")
    val spark = new SparkContext(conf)

    val words = spark
      .wholeTextFiles(path)
      .flatMap { case (fileName, text) => text.split("[^\\w\\d]") }
      .map(word => (word, 1))
      .reduceByKey(_ + _)
      .sortBy(tuple => tuple._2, false)
      .collect()

    spark.stop()
  }
}