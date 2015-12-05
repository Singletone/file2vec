import org.apache.spark.{SparkContext, SparkConf}

abstract class ProcessingContext(pathTo: PathTo, hyperParams: HyperParameters) {
  val sparkConf = new SparkConf().setAppName("Data processor").setMaster("local[4]")
  val spark = new SparkContext(sparkConf)

  def process() = {
    this.extract()
    this.clean()
    this.describe()
    this.prune()
    this.transform()
  }

  def extract()

  def clean() = {

  }

  def describe() = {

  }

  def prune() = {

  }

  def transform() = {

  }
}
