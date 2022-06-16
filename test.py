import sqlalchemy
import os
import base64
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
with open("encoder.pickle", "rb") as image_file:
    encoder = base64.b64encode(image_file.read())
print("ENCODER:")
print(encoder)
with db.connect() as conn:
    stats = pd.read_sql(
        "select * from model where selected = 1", conn)
    print("MINE:")
    try:
        with open("q1.txt", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 12], encoding='utf-8')))
    except Exception as e:
        print(str(e))
