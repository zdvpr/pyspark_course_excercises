import pytest
from pyspark.sql import SparkSession
from soda.scan import Scan

from pyspark.sql.functions import *
from pyspark.sql.types import *

@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder \
      .master("local") \
      .appName("chispa") \
      .getOrCreate()


def build_scan(name, spark_session):
    scan = Scan()
    scan.disable_telemetry()
    scan.set_scan_definition_name("data_quality_test")
    scan.set_data_source_name("spark_df")
    scan.add_spark_session(spark_session)
    return scan

def test_videos_source(spark):
    videos_df = spark.read.option('header', 'true').option("inferSchema", "true").csv('datasets/USvideos.csv')
    videos_df.createOrReplaceTempView('videos')

    scan = build_scan("videos_source_data_quality_test", spark)
    scan.add_sodacl_yaml_file("data_quality/videos_checks.yml")

    scan.execute()

    scan.assert_no_checks_warn_or_fail()

comments_schema = StructType([ \
    StructField("video_id", StringType(), True), \
    StructField("comment_text", StringType(), True), \
    StructField("likes", IntegerType(), False), \
    StructField("replies", IntegerType(), False)])

def test_broken_comments(spark):
    comments_schema_w_corr = StructType(comments_schema.fields + [StructField("corrupt_record", StringType(),True)] )
    comments = spark.read.option('header', 'true').option("mode", "PERMISSIVE").option("columnNameOfCorruptRecord", "corrupt_record").schema(comments_schema_w_corr).csv('datasets/UScomments.csv')

    cnt_all = comments.count()
    cnt_correct = comments.filter(isnull(col("corrupt_record")) ).count()

    if cnt_correct / cnt_all >= 0.5:
        assert True
    else:
        assert False

def test_schema_comments(spark):
    comments = spark.read.option('header', 'true').option("mode", "DROPMALFORMED").csv('datasets/UScomments.csv')
    assert comments.schema == comments_schema

def test_likes_replies_comments(spark):
    comments = spark.read.option('header', 'true').option("mode", "DROPMALFORMED").csv('datasets/UScomments.csv')
    less_zero_cnt  = comments.filter((col("likes")<0) | (col("replies")<0) ).count()
    likes_great_zero_cnt  = comments.filter( col("likes")>0 ).count()
    replies_great_zero_cnt = comments.filter( col("replies") > 0 ).count()

    assert less_zero_cnt == 0 and likes_great_zero_cnt > 0 and replies_great_zero_cnt > 0

def test_video_id_comments(spark):
    comments = spark.read.option('header', 'true').option("mode", "DROPMALFORMED").csv('datasets/UScomments.csv')
    wrong_id_cnt = comments.select(regexp_extract(col("video_id"), r"[^\w\d\-\_]+",0).alias("wrong_symbols")).filter(length("wrong_symbols")>0).count()

    assert wrong_id_cnt == 0