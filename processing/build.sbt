name := "processing"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.2",
  "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test"
)
