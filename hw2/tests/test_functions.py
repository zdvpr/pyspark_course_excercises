import pytest
from pyspark.sql import SparkSession
from chispa import *

from pyspark.sql.functions import *
from pyspark.sql.types import *
import video_analytics.functions as F

@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder \
      .master("local") \
      .appName("chispa") \
      .getOrCreate()

def test_score_calculation(spark):
    data = [
        (10, 4, 2, 3, 1, 27 ),
        (27, 100, 30, 50, 70, 597)]

    df = spark.createDataFrame(data, ["views", "likes", "dislikes", "likes_cnt", "comments_cnt", "expected_value"])

    df = df.withColumn("score", round(F.score_calculation("views", "likes", "dislikes", "likes_cnt", "comments_cnt"), 0).cast(IntegerType()))

    assert_column_equality(df, "score", "expected_value")

def test_median(spark):
    data = [
        ('show', 1 ),
        ('show', 10),
        ('show', 5),
        ('show', 14),
        ('show', 100),
        ('animals', 5),
        ('animals', 7),
        ('animals', 9)]

    schema_exp = StructType([ \
        StructField("category", StringType(), True), \
        StructField("score", IntegerType(), True)])

    expected =  [('animals', 7),
                 ('show', 10)]

    df_input = spark.createDataFrame(data, ["category", "score"])
    df_expected = spark.createDataFrame(expected, schema_exp)
    df_result = df_input.groupBy('category').applyInPandas(F.median_udf, "category string, score int")

    assert_df_equality(df_result, df_expected)

def test_split_string(spark):
    data = [
        ('awe| fd|wer 23', ['awe',' fd','wer 23'] ),
        ('# % $ |a bc', ['# % $ ','a bc'])]

    expected =  [('awe| fd|wer 23', 'awe'),
                 ('awe| fd|wer 23', ' fd'),
                 ('awe| fd|wer 23', 'wer 23'),
                 ('# % $ |a bc', '# % $ '),
                 ('# % $ |a bc', 'a bc')]

    df = spark.createDataFrame(data, ["input"])
    df = df.withColumn("split_result", F.split_string("input")).select("input",explode("split_result").alias("split_result"))
    df_expected = spark.createDataFrame(expected, ["input", "split_result"])
    assert_df_equality(df, df_expected)
