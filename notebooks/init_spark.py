from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


def start_spark(app_name="Credit Risk Modeling"):
    """
    Initializes and returns a SparkSession configured for local development
    with Delta Lake support and 4-core parallelism.

    Parameters:
    ----------
    app_name : str
        Name for the Spark application (shown in Spark UI)

    Returns:
    -------
    SparkSession
    """

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.warehouse.dir", "./data/spark-warehouse")
    )  # Consistent location for managed tables

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    
    print(spark.version)
    
    return spark

if __name__ == "__main__":
    spark = start_spark()
