import sqlalchemy
import os
import pandas as pd


def init_db_connection():
    db_config = {
        'pool_size': 5,
        'max_overflow': 2,
        'pool_timeout': 30,
        'pool_recycle': 1800,
    }
    return init_unix_connection_engine(db_config)


def init_unix_connection_engine(db_config):
    # Dev
    pool = sqlalchemy.create_engine(
        sqlalchemy.engine.url.URL(
            host="127.0.0.1",
            port="5432",
            drivername="postgres+pg8000",
            username="postgres",
            password="Password123ajjw",
            database="mycareersfuture",
        ),
        **db_config
    )

    pool.dialect.description_encoding = None
    return pool


db = init_db_connection()

with db.connect() as conn:
    stats = pd.read_sql(
        "select Date(\"createdDate\") as day, min_rmse, min_rsquare, max_rmse, max_rsquare from model order by 1 desc limit 5", conn)
    print([{"name": str(x), "Min R² Square Value":  round(y, 3), "Max R² Square Value": round(z, 3)}
          for x, y, z in zip(stats["day"], stats["min_rsquare"], stats["max_rsquare"])])
    print([{"name": str(x), "Min RMSE":  round(y, 2), "Max RMSE": round(z, 2)}
          for x, y, z in zip(stats["day"], stats["min_rmse"], stats["max_rmse"])])
    stats = pd.read_sql(
        "select Date(crawldate) as day, count(*) as count from careers group by DATE(crawldate) order by 1 desc limit 5;", conn)
    print([{"name": str(x), "New Job":  round(y, 2)}
          for x, y in zip(stats["day"], stats["count"])])
