import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *

@pandas_udf(DoubleType())
def score_calculation( views: pd.Series , likes: pd.Series , dislikes: pd.Series , likes_cnt: pd.Series , comments_cnt: pd.Series ) ->  pd.Series :
    return (views + 2 * (likes - dislikes) + 3 * likes_cnt + 4 * comments_cnt)

def median_udf(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df.groupby(["category"])["score"].median()).reset_index()

@pandas_udf(ArrayType(StringType()))
def split_string(column: pd.Series) -> pd.Series:
    return column.str.split("|")