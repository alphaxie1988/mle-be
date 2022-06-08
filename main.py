#   ___ __  __ ___  ___  ___ _____
#  |_ _|  \/  | _ \/ _ \| _ \_   _|
#   | || |\/| |  _/ (_) |   / | |
#  |___|_|  |_|_|  \___/|_|_\ |_|

import random
import datetime
import logging
import os
import time
import requests
from flask import Flask, render_template, request, Response
import sqlalchemy
import json
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import requests
from flask_cors import CORS
from dotenv import load_dotenv

#   ___ _  _ ___ _____ ___   _   _    ___ ___ ___ _  _  ___
#  |_ _| \| |_ _|_   _|_ _| /_\ | |  |_ _/ __|_ _| \| |/ __|
#   | || .` || |  | |  | | / _ \| |__ | |\__ \| || .` | (_ |
#  |___|_|\_|___| |_| |___/_/ \_\____|___|___/___|_|\_|\___|
load_dotenv()
app = Flask(__name__)
CORS(app)
logger = logging.getLogger()

#  ___ _   _ _  _  ___ _____ ___ ___  _  _
#  | __| | | | \| |/ __|_   _|_ _/ _ \| \| |
#  | _|| |_| | .` | (__  | |  | | (_) | .` |
#  |_|  \___/|_|\_|\___| |_| |___\___/|_|\_|


def insert_varibles_into_table(conn, uuid, title, description, minimumYearsExperience, skills, numberOfVacancies, categories, employmentTypes, positionLevels, totalNumberOfView, totalNumberJobApplication, originalPostingDate, expiryDate, links, postedCompany, minsalary, maxsalary, avgsalary):
    # 2)insert into postgres mycareerfuture table
    PSql_insert_query = """INSERT INTO careers(uuid,title,description,minimumYearsExperience,skills,numberOfVacancies,categories,employmentTypes,positionLevels,totalNumberOfView,totalNumberJobApplication,originalPostingDate,expiryDate,links,postedCompany,minsalary,maxsalary,avgsalary,crawldate,status,remarks)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),0,'')"""

    record = (uuid, title, description, minimumYearsExperience, skills, numberOfVacancies, categories, employmentTypes, positionLevels,
              totalNumberOfView, totalNumberJobApplication, originalPostingDate, expiryDate, links, postedCompany, minsalary, maxsalary, avgsalary)
    conn.execute(PSql_insert_query, record)
    # connection.commit()
    print(".", end="", flush=True)


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
                username="postgres",
                password="Password123ajjw",
                database="mycareersfuture",
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

#   ___ ___ ___ _____ ___ _   _ _        _   ___ ___
#  | _ \ __/ __|_   _| __| | | | |      /_\ | _ \_ _|
#  |   / _|\__ \ | | | _|| |_| | |__   / _ \|  _/| |
#  |_|_\___|___/ |_| |_|  \___/|____| /_/ \_\_| |___|


@app.route('/', methods=['GET'])
def index():
    # votes = []
    # with db.connect() as conn:
    #     # Execute the query and fetch all results
    #     recent_votes = conn.execute(
    #         "SELECT * from mle"
    #     ).fetchall()
    #     # Convert the results into a list of dicts representing votes
    #     for row in recent_votes:
    #         votes.append({
    #             'candidate': row[0],
    #             'time_cast': row[1]
    #         })

    #     stmt = sqlalchemy.text(
    #         "SELECT num_votes FROM totals WHERE candidate=:candidate")
    #     # Count number of votes for tabs
    #     tab_result = conn.execute(stmt, candidate="TABS").fetchone()
    #     tab_count = tab_result[0] if tab_result is not None else 0
    #     # Count number of votes for spaces
    #     space_result = conn.execute(stmt, candidate="SPACES").fetchone()
    #     space_count = space_result[0] if space_result is not None else 0

    # return render_template(
    #     'index.html',
    #     recent_votes=votes,
    #     tab_count=tab_count,
    #     space_count=space_count
    # )
    return Response(
        status=200,
        response="Up and Running: Student Project by SMU"
    )


@app.route('/', methods=['POST'])
def save_vote():
    # Get the team and time the vote was cast.
    team = request.form['team']
    time_cast = datetime.utcnow()
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
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Crawl%20Started%20at%20"+str(datetime.now())[0:-7])

    # Start to Crawl Code Here
    print("Start to Crawl")
    with db.connect() as conn:
        conn.execute("UPDATE mle SET value = '1' WHERE key='crawling'")
    ##################### JIE YUAN START HERE ##############
    with db.connect() as conn:
        numberOfJobBefore = int(conn.execute(
            "SELECT count(*) FROM careers").fetchone()[0])
    # 1)crawl from careersgfuture order by posted dated if full crawl ie(date is null), crawl everything else crawl until date
    try:
        with db.connect() as conn:

            ####### end of connection ####
            max_date = pd.read_sql(
                "select max(careers.originalpostingdate) from careers", conn).iloc[0, 0]
            print("Max Date:", max_date)

            if max_date is not None:
                #  _  _   _   _    ___    ___ ___    ___      ___
                # | || | /_\ | |  | __|  / __| _ \  /_\ \    / / |
                # | __ |/ _ \| |__| _|  | (__|   / / _ \ \/\/ /| |__
                # |_||_/_/ \_\____|_|    \___|_|_\/_/ \_\_/\_/ |____|

                print("Start Crawling from", max_date)
                max_date_uuid = pd.read_sql(
                    "select uuid from careers where originalpostingdate='" + str(max_date) + "'", conn)
                max_date_uuid_list = max_date_uuid['uuid'].tolist()

                for page in range(1000):
                    r = requests.get(
                        "https://api.mycareersfuture.gov.sg/v2/jobs?limit=100&page=" + str(page) + "&sortBy=new_posting_date")
                    if (r.status_code == 200):
                        result = json.loads(r.text)
                        if len(result["results"]) != 0:
                            # check 1st date of each page extracted. when latest updated date < max date, the loop will stop
                            latest_post_date_1st_row = datetime.strptime(
                                result["results"][0]["metadata"]["newPostingDate"], "%Y-%m-%d").date()

                            if latest_post_date_1st_row < max_date:
                                break

                            for res in result["results"]:
                                org_post_date = datetime.strptime(
                                    res["metadata"]["originalPostingDate"], "%Y-%m-%d").date()

                                if org_post_date > max_date:
                                    try:
                                        insert_varibles_into_table(conn, res["uuid"], res["title"], BeautifulSoup(res["description"].replace("\n", " "), 'html.parser').get_text(), res["minimumYearsExperience"], "|".join([x["skill"] for x in res["skills"]]), res["numberOfVacancies"], "|".join([x["category"] for x in res["categories"]]), "|".join([x["employmentType"] for x in res["employmentTypes"]]), "|".join(
                                            [x["position"] for x in res["positionLevels"]]), res["metadata"]["totalNumberOfView"], res["metadata"]["totalNumberJobApplication"], res["metadata"]["originalPostingDate"], res["metadata"]["expiryDate"], res["_links"]["self"]["href"], res["postedCompany"]["name"], res["salary"]["minimum"], res["salary"]["maximum"], int((res["salary"]["maximum"]+res["salary"]["minimum"]) / 2))
                                    except Exception:
                                        pass
                                elif org_post_date == max_date:
                                    if res["uuid"] not in max_date_uuid_list:
                                        try:
                                            insert_varibles_into_table(conn, res["uuid"], res["title"], BeautifulSoup(res["description"].replace("\n", " "), 'html.parser').get_text(), res["minimumYearsExperience"], "|".join([x["skill"] for x in res["skills"]]), res["numberOfVacancies"], "|".join([x["category"] for x in res["categories"]]), "|".join([x["employmentType"] for x in res["employmentTypes"]]), "|".join(
                                                [x["position"] for x in res["positionLevels"]]), res["metadata"]["totalNumberOfView"], res["metadata"]["totalNumberJobApplication"], res["metadata"]["originalPostingDate"], res["metadata"]["expiryDate"], res["_links"]["self"]["href"], res["postedCompany"]["name"], res["salary"]["minimum"], res["salary"]["maximum"], int((res["salary"]["maximum"]+res["salary"]["minimum"]) / 2))
                                        except Exception:
                                            pass
                                    else:
                                        continue
                                else:
                                    break
                        else:
                            break

            else:
                #  ___ _   _ _    _       ___ ___    ___      ___
                # | __| | | | |  | |     / __| _ \  /_\ \    / / |
                # | _|| |_| | |__| |__  | (__|   / / _ \ \/\/ /| |__
                # |_|  \___/|____|____|  \___|_|_\/_/ \_\_/\_/ |____|
                print("Start Full Crawl")
                for page in range(2000):
                    r = requests.get(
                        "https://api.mycareersfuture.gov.sg/v2/jobs?limit=100&page=" + str(page) + "&sortBy=new_posting_date")
                    if (r.status_code == 200):
                        result = json.loads(r.text)
                        if len(result["results"]) != 0:
                            for res in result["results"]:
                                try:
                                    insert_varibles_into_table(conn, res["uuid"], res["title"], BeautifulSoup(res["description"].replace("\n", " "), 'html.parser').get_text(), res["minimumYearsExperience"], "|".join([x["skill"] for x in res["skills"]]), res["numberOfVacancies"], "|".join([x["category"] for x in res["categories"]]), "|".join([x["employmentType"] for x in res["employmentTypes"]]), "|".join(
                                        [x["position"] for x in res["positionLevels"]]), res["metadata"]["totalNumberOfView"], res["metadata"]["totalNumberJobApplication"], res["metadata"]["originalPostingDate"], res["metadata"]["expiryDate"], res["_links"]["self"]["href"], res["postedCompany"]["name"], res["salary"]["minimum"], res["salary"]["maximum"], int((res["salary"]["maximum"]+res["salary"]["minimum"]) / 2))
                                except Exception as e:
                                    print("Error while insert:", e)
                                    pass
                        else:
                            break

    except Exception as e:
        logger.exception(e)
        return Response(
            status=500,
            response="Unable to Crawl Propoerly! Please check the "
            "application logs for more details."
        )
    finally:
        if (conn):
            conn.close()
            print("PostgreSQL connection is closed")
    ###################### JIE YUAN END HERE ###############
    # 3)trigger cleaning
    clean()
    # 4) update database to say crawling ended
    with db.connect() as conn:
        conn.execute("UPDATE mle SET value = '0' WHERE key='crawling'")
    print("Crawl Finish")

    # 5) Count number of new job
    with db.connect() as conn:
        numberOfJobAfter = int(conn.execute(
            "SELECT count(*) FROM careers").fetchone()[0])
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Crawl%20Ended%20at%20"+str(datetime.now())[0:-7]+"%0ANumber%20Of%20New%20Jobs:%20"+str(numberOfJobAfter-numberOfJobBefore))
    # End of Crawl
    return f"Thank you for waiting!"
    # return Response(
    # status=200,
    # response="Thank you for waiting!"
    # )


def clean():
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Cleaning%20Started%20at%20"+str(datetime.now())[0:-7])
    ########### Start Cleaning ############
    ######## Anna start here ###########
    # 1) read all data from careers table where column included is null
    # 2) Detect outlier update column remark=""
    # 3) Spell check or any other potential problem
    # 4) update column included = included

    # Salary too wide
    with db.connect() as conn:
        numberOfFlagedBefore = int(conn.execute(
            "SELECT count(*) FROM careers where status='1'").fetchone()[0])
    with db.connect() as conn:
        conn.execute(
            "update careers set status = 1, remarks=Concat(remarks,'Salary Range too Wide|') where maxsalary - minsalary > 15000 and status = 0")
    with db.connect() as conn:
        conn.execute(
            "update careers set status = 1, remarks=Concat(remarks,'Min salary is too low|') where minsalary <= 100 and status = 0")
    with db.connect() as conn:
        conn.execute(
            "update careers set status = 1, remarks=Concat(remarks,'Max salary is too hight|') where maxsalary >= 50000 and status = 0")
    # update the rest
    with db.connect() as conn:
        conn.execute("update careers set status = 2 where status = 0")
    with db.connect() as conn:
        numberOfFlagedAfter = int(conn.execute(
            "SELECT count(*) FROM careers where status='1'").fetchone()[0])

    # 0 - crawl not check
    # 1 - checked and flagged
    # 2 - checked and notflagged [use for training]
    # 3 - checked and flagged and verified [use for training]
    # 4 - checked and flagged and verified not use for training [not use for training]

    ######## Anna end here #########
    ########## End Cleaning ##############
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Cleaning%20Ended%20at%20"+str(datetime.now())[0:-7]+"%0ANumber%20Of%20Flagged:%20"+str(numberOfFlagedAfter-numberOfFlagedBefore))
    train()


def train():
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Training%20Started%20at%20"+str(datetime.now())[0:-7])
    time.sleep(15)
    # select * from careers where error is not null and fixed = "included"
    # fixed can be null -> yet to fixed, fixed => excluded, fixed => included
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Training%20Ended%20at%20"+str(datetime.now())[0:-7])


@app.route("/predict")
def predict():
    # model = pickle.load("gs://sadsaas/asd.model")
    # return model.predict(job)
    return Response(json.dumps({"pMinSal": random.randint(1000, 20000), "pMaxSal": random.randint(1000, 20000)}),  mimetype='application/json')


@app.route("/stats")
def stats():
    rsquarevalue = [{"name": str(
        x+1)+" Jun 2022",  "R Square Value":  random.randint(80, 100)} for x in range(8)]
    RSME = [{"name": str(
        x+1)+" Jun 2022",  "RSME":  random.randint(800, 1000)/1000} for x in range(8)]
    newjob = [{"name": str(
        x+1)+" Jun 2022",  "New Job":  random.randint(140, 200)} for x in range(8)]
    return Response(json.dumps({"rsquarevalue": rsquarevalue, "RSME": RSME, "newjob": newjob}), 200, mimetype='application/json')


@app.route("/outlier")
def data():
    df = pd.read_sql(
        "select uuid, title, left(description,50) as description, skills, numberofvacancies, categories, positionlevels, postedcompany,employmenttypes, minsalary, maxsalary , remarks from careers where status = 1", db.connect())

    return Response(json.dumps([{v: x[k] for (k, v) in enumerate(df)}for x in df.values]
                               ), 200,  mimetype='application/json')


@app.route('/outlier', methods=['PUT'])
def updateData():
    print(request.get_json()['action'])
    print(str(request.get_json()['payload'])[1:-1])
    if request.get_json()['action'] == "Add":
        with db.connect() as conn:
            conn.execute(
                "update careers set status = 3, remarks=Concat(remarks,'Admin marked as OK|') where uuid in ("+str(request.get_json()['payload'])[1:-1]+")")
    elif request.get_json()['action'] == "Hide":
        with db.connect() as conn:
            conn.execute(
                "update careers set status = 4, remarks=Concat(remarks,'Admin marked as not OK|') where uuid in ("+str(request.get_json()['payload'])[1:-1]+")")
    return Response(json.dumps({"success": True}), 200,  mimetype='application/json')


# Set Crawling = False when starting
with db.connect() as conn:
    conn.execute("UPDATE mle SET value = '0' WHERE key='crawling'")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

# ASCII ART FROM: https://patorjk.com/software/taag/#p=display&f=Small&t=HALF%20CRAWL
