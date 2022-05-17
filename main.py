import datetime
import logging
import os
import time
import requests
from flask import Flask, render_template, request, Response
import sqlalchemy

# Hydrate the environment from the .env file
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

logger = logging.getLogger()


def init_db_connection():
    db_config = {
        'pool_size': 5,
        'max_overflow': 2,
        'pool_timeout': 30,
        'pool_recycle': 1800,
    }
    return init_unix_connection_engine(db_config)


def init_unix_connection_engine(db_config):
    if(os.environ.get('DB_PASS') == None):
        # Dev
        pool = sqlalchemy.create_engine(
            sqlalchemy.engine.url.URL(
                host="127.0.0.1",
                port="5432",
                drivername="postgres+pg8000",
                username=os.environ.get('DB_USER'),
                password="Password123ajjw",
                database=os.environ.get('DB_NAME'),
            ),
            **db_config
        )
    else:
        # Production
        pool = sqlalchemy.create_engine(
            sqlalchemy.engine.url.URL(
                drivername="postgres+pg8000",
                username=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASS'),
                database=os.environ.get('DB_NAME'),
                query={
                    "unix_sock": "/cloudsql/{}/.s.PGSQL.5432".format(
                        os.environ.get('CLOUD_SQL_CONNECTION_NAME')),
                }
            ),
            **db_config
        )
    pool.dialect.description_encoding = None
    return pool


db = init_db_connection()


@app.route('/', methods=['GET'])
def index():
    votes = []
    with db.connect() as conn:
        # Execute the query and fetch all results
        recent_votes = conn.execute(
            "SELECT candidate, time_cast FROM votes "
            "ORDER BY time_cast DESC LIMIT 5"
        ).fetchall()
        # Convert the results into a list of dicts representing votes
        for row in recent_votes:
            votes.append({
                'candidate': row[0],
                'time_cast': row[1]
            })

        stmt = sqlalchemy.text(
            "SELECT num_votes FROM totals WHERE candidate=:candidate")
        # Count number of votes for tabs
        tab_result = conn.execute(stmt, candidate="TABS").fetchone()
        tab_count = tab_result[0] if tab_result is not None else 0
        # Count number of votes for spaces
        space_result = conn.execute(stmt, candidate="SPACES").fetchone()
        space_count = space_result[0] if space_result is not None else 0

    return render_template(
        'index.html',
        recent_votes=votes,
        tab_count=tab_count,
        space_count=space_count
    )


@app.route('/', methods=['POST'])
def save_vote():
    # Get the team and time the vote was cast.
    team = request.form['team']
    time_cast = datetime.datetime.utcnow()
    # Verify that the team is one of the allowed options
    if team != "TABS" and team != "SPACES":
        logger.warning(team)
        return Response(
            response="Invalid team specified.",
            status=400
        )

    stmt = sqlalchemy.text(
        "INSERT INTO votes (time_cast, candidate)"
        " VALUES (:time_cast, :candidate)"
    )
    totals_stmt = sqlalchemy.text(
        "UPDATE totals SET num_votes = num_votes + 1 WHERE candidate=:candidate"
    )
    try:
        with db.connect() as conn:
            conn.execute(stmt, time_cast=time_cast, candidate=team)
            conn.execute(totals_stmt, candidate=team)
    except Exception as e:
        logger.exception(e)
        return Response(
            status=500,
            response="Unable to successfully cast vote! Please check the "
                     "application logs for more details."
        )

    return Response(
        status=200,
        response="Vote successfully cast for '{}' at time {}!".format(
            team, time_cast)
    )


@app.route("/crawl")
def crawl():
    # Check if system is crawling
    with db.connect() as conn:
        isCrawling = conn.execute(
            "SELECT value FROM mle where key='crawling'").fetchone()
        isCrawling = bool(int(isCrawling[0]))
    if (isCrawling):
        return f"Already Crawling, please wait!"
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Crawl%20Started%20at%20"+str(datetime.datetime.now()))

    # Start to Crawl Code Here
    print("Start to Crawl")
    with db.connect() as conn:
        conn.execute("UPDATE mle SET value = '1' WHERE key='crawling'")
    ##################### JIE YUAN START HERE ##############

    fullcrawl = request.args.get('fullcrawl')
    # 1)crawl from careersgfuture order by posted dated if full crawl ie(date is null), crawl everything else crawl until date
    # sleep 15 to simulate we use 15 second to crawl
    time.sleep(15)
    # 2)insert into postgres mycareerfuture table

    if(fullcrawl):
        return f"Start Full Crawl {fullcrawl}!"

    ###################### JIE YUAN END HERE ###############
    # 3)trigger cleaning
    clean()
    with db.connect() as conn:
        conn.execute("UPDATE mle SET value = '0' WHERE key='crawling'")
    print("Crawl Finish")
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Crawl%20Ended%20at%20"+str(datetime.datetime.now()))
    # End of Crawl
    return f"Thank you for waiting!"


def clean():
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Cleaning%20Started%20at%20"+str(datetime.datetime.now()))
    ########### Start Cleaning ############
    ######## Anna start here ###########
    # 1) read all data from careers table where column included is null
    # 2) Detect outlier update column remark=""
    # 3) Spell check or any other potential problem
    # 4) update column included = included
    time.sleep(15)
    ######## Anna end here #########
    ########## End Cleaning ##############
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Cleaning%20Ended%20at%20"+str(datetime.datetime.now()))


def train():
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Training%20Started%20at%20"+str(datetime.datetime.now()))

    # select * from careers where error is not null and fixed = "included"
    # fixed can be null -> yet to fixed, fixed => excluded, fixed => included
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
