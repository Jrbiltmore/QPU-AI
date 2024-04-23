# /Data_Handling_Processing/big_data_config.py

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

def create_spark_session(app_name="BigDataApplication", master="local[*]"):
    """
    Create a Spark session.

    Parameters:
    - app_name (str): Name of the Spark application (default="BigDataApplication").
    - master (str): URL of the Spark master (default="local[*]").

    Returns:
    - SparkSession: Spark session object.
    """
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
    """
    Read data from a file using Spark session.

    Parameters:
    - spark (SparkSession): Spark session object.
    - file_path (str): Path to the data file.

    Returns:
    - DataFrame: DataFrame containing the read data.
    """
    # Read data using Spark session
    return spark.read.csv(file_path, header=True, inferSchema=True)

def aggregate_data(df, group_by_cols, agg_funcs):
    """
    Aggregate data using specified group by columns and aggregation functions.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - group_by_cols (list): List of columns to group by.
    - agg_funcs (dict): Dictionary of aggregation functions {column: function}.

    Returns:
    - DataFrame: Aggregated DataFrame.
    """
    # Perform data aggregation
    agg_df = df.groupby(group_by_cols).agg(agg_funcs)
    return agg_df

def join_data(df1, df2, join_type="inner", on=None):
    """
    Join two DataFrames.

    Parameters:
    - df1 (DataFrame): First DataFrame.
    - df2 (DataFrame): Second DataFrame.
    - join_type (str): Type of join (default="inner").
    - on (str or list): Column(s) to join on.

    Returns:
    - DataFrame: Joined DataFrame.
    """
    # Perform DataFrame join
    joined_df = df1.join(df2, on=on, how=join_type)
    return joined_df

# Example usage
if __name__ == '__main__':
    # Create Spark session
    spark = create_spark_session()

    # Read data
    df = read_data(spark, 'path_to_your_data.csv')

    # Read additional data
    df2 = read_data(spark, 'path_to_additional_data.csv')

    # Perform join
    joined_df = join_data(df, df2, join_type="inner", on="key_column")

    # Show joined data
    joined_df.show()

    # Perform aggregation
    agg_df = aggregate_data(joined_df, group_by_cols=["group_column"], agg_funcs={"value_column": "sum"})

    # Show aggregated data
    agg_df.show()
