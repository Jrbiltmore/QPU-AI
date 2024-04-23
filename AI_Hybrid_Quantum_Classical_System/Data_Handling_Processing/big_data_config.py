
# /Data_Handling_Processing/big_data_config.py

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

def create_spark_session(app_name="BigDataApplication", master="local[*]"):
    # Setup Spark configuration
    conf = SparkConf().setAppName(app_name).setMaster(master)
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "2g")
    conf.set("spark.executor.cores", "4")

    # Create Spark context and session
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    return spark

def read_data(spark, file_path):
    # Read data using Spark session
    return spark.read.csv(file_path, header=True, inferSchema=True)

# Example usage
if __name__ == '__main__':
    spark = create_spark_session()
    df = read_data(spark, 'path_to_your_data.csv')
    df.show()
