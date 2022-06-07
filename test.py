import sqlalchemy
import os


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

db.connect().execute("insert into mle values ('apple','5')")
