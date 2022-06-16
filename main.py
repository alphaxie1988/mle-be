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

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import feature_extraction
from sklearn import model_selection
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
import keras
import pickle
from numpy.random import seed
import tensorflow as tf
import base64

#   ___ _  _ ___ _____ ___   _   _    ___ ___ ___ _  _  ___
#  |_ _| \| |_ _|_   _|_ _| /_\ | |  |_ _/ __|_ _| \| |/ __|
#   | || .` || |  | |  | | / _ \| |__ | |\__ \| || .` | (_ |
#  |___|_|\_|___| |_| |___/_/ \_\____|___|___/___|_|\_|\___|
load_dotenv()
app = Flask(__name__)
CORS(app)
logger = logging.getLogger()

#   ___ _   _ _  _  ___ _____ ___ ___  _  _
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
        'pool_size': 10,
        'max_overflow': 4,
        'pool_timeout': 300,
        'pool_recycle': 3600,
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


@app.route("/resetcrawl")
def resetcrawl():
    with db.connect() as conn:
        conn.execute("update mle set value = 0 where key = 'crawling'")
    return Response(
        status=200,
        response="Reset success")


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
                                        print("x", end="", flush=True)
                                        pass
                                elif org_post_date == max_date:
                                    if res["uuid"] not in max_date_uuid_list:
                                        try:
                                            insert_varibles_into_table(conn, res["uuid"], res["title"], BeautifulSoup(res["description"].replace("\n", " "), 'html.parser').get_text(), res["minimumYearsExperience"], "|".join([x["skill"] for x in res["skills"]]), res["numberOfVacancies"], "|".join([x["category"] for x in res["categories"]]), "|".join([x["employmentType"] for x in res["employmentTypes"]]), "|".join(
                                                [x["position"] for x in res["positionLevels"]]), res["metadata"]["totalNumberOfView"], res["metadata"]["totalNumberJobApplication"], res["metadata"]["originalPostingDate"], res["metadata"]["expiryDate"], res["_links"]["self"]["href"], res["postedCompany"]["name"], res["salary"]["minimum"], res["salary"]["maximum"], int((res["salary"]["maximum"]+res["salary"]["minimum"]) / 2))
                                        except Exception:
                                            print("x", end="", flush=True)
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
                                    print("x", end="", flush=True)
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


@app.route("/train")
def train():
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Training%20Started%20at%20"+str(datetime.now())[0:-7])
    # time.sleep(15)

    seed(2021)
    tf.random.set_seed(2021)

    with db.connect() as conn:
        train_days = pd.read_sql(
            "select value from mle where key = 'numOfDaysToTrain'", conn)
        print("select * from careers where originalpostingdate >= current_date - INTERVAL '" +
              str(train_days.iloc[0, 0]) + " day and status in (2,3)")
        df_raw = pd.read_sql("select * from careers where originalpostingdate >= current_date - INTERVAL '" +
                             str(train_days.iloc[0, 0]) + " day' and status in (2,3)", conn)

    # df_main = df_raw.head(1000)
    df_main = df_raw.copy()
    df_main = df_main[df_main["avgsalary"] < 15000]
    df_main = df_main[df_main["minimumyearsexperience"] <= 30]
    df_main = df_main[df_main["avgsalary"] > 1500]
    df_main.fillna(value='', inplace=True)
    print(df_main.info())
    df_main['categories'] = df_main['categories'].str.replace(' ', '_')
    df_main['categories'] = df_main['categories'].str.replace('|', ' ')
    df_main['employmenttypes'] = df_main['employmenttypes'].str.replace(
        ' ', '_')
    df_main['employmenttypes'] = df_main['employmenttypes'].str.replace(
        '|', ' ')
    df_main['new_col'] = df_main[['categories', 'employmenttypes']].apply(
        lambda x: '|'.join(x.dropna().values.tolist()), axis=1)

    # x_train, x_test, y_train, y_test = model_selection.train_test_split(df_main.loc[:, df_main.columns != 'avgsalary'], df_main["avgsalary"], test_size = 0.2, random_state = 2021)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(df_main.loc[:, ~(df_main.columns.isin(
        ['avgsalary', 'minsalary', 'maxsalary']))], df_main[["minsalary", "maxsalary"]], test_size=0.2, random_state=2021)

    # Word vectorizer
    # min_df = 0.01, max_df = 0.5, stop_words = 'english'
    count_vectorizer = feature_extraction.text.CountVectorizer()
    # fit dont put fit into test - fit mean you want to fix the module
    x_train_categories = count_vectorizer.fit_transform(x_train["new_col"])
    x_test_categories = count_vectorizer.transform(x_test["new_col"])
    x_test_categories_df = pd.DataFrame(
        x_test_categories.todense(), columns=count_vectorizer.get_feature_names())
    x_train_categories_df = pd.DataFrame(
        x_train_categories.todense(), columns=count_vectorizer.get_feature_names())

    x_train = pd.concat([x_train_categories_df, x_train[['minimumyearsexperience',
                        'numberofvacancies', 'positionlevels']].reset_index(drop=True), ], axis=1)
    x_test = pd.concat([x_test_categories_df, x_test[['minimumyearsexperience', 'numberofvacancies',
                       'positionlevels']].reset_index(drop=True), ], axis=1)

    # Prepare HotEncoder - To change categorical into 1,0
    enc = OneHotEncoder(handle_unknown='ignore')

    # This are column that are categorical
    categorical = ['positionlevels']
    enc.fit(x_train[categorical])
    feature_name = enc.get_feature_names(x_train[categorical].columns)
    x_train_one_hot_data = enc.fit_transform(x_train[categorical]).toarray()
    x_test_one_hot_data = enc.transform(x_test[categorical]).toarray()

    x_train_one_hot_data_df = pd.DataFrame(
        x_train_one_hot_data, columns=feature_name)
    x_test_one_hot_data_df = pd.DataFrame(
        x_test_one_hot_data, columns=feature_name)
    x_train = pd.concat([x_train_categories_df, x_train_one_hot_data_df, x_train[[
                        'minimumyearsexperience', 'numberofvacancies']].reset_index(drop=True), ], axis=1)
    x_test = pd.concat([x_test_categories_df, x_test_one_hot_data_df, x_test[[
                       'minimumyearsexperience', 'numberofvacancies']].reset_index(drop=True), ], axis=1)
    print(x_train.shape)
    type(x_train)

    nn = Sequential()
    nn.add(Dense(78, input_dim=x_train.shape[1], activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Dense(39, activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Dense(19, activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Dense(10, activation='relu'))
    nn.add(Dropout(0.2))
    # nn.add(layers.Dense(1, activation=''))
    nn.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    nn.compile(loss='mean_squared_error', optimizer='adam')
    # nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(x_test.shape)

    # minsalary model
    nn.fit(x_train, y_train['minsalary'], epochs=20, batch_size=100, verbose=2)
    y_pred_test_nn_min = nn.predict(x_test)

    min_MSE = mean_squared_error(y_test['minsalary'], y_pred_test_nn_min)
    min_RMSE = mean_squared_error(
        y_test['minsalary'], y_pred_test_nn_min, squared=False)
    min_R2 = r2_score(y_test['minsalary'], y_pred_test_nn_min)
    min_adj_R2 = 1-(1-r2_score(y_test['minsalary'], y_pred_test_nn_min))*(
        (x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1))

    print('Min salary MSE: ' + str(min_MSE))
    print('Min salary RMSE: ' + str(min_RMSE))
    print('Min salary R2:' + str(min_R2))
    print('Min salary Adjusted R2: ' + str(min_adj_R2))
    nn.save("model_min.h5")
    with open("model_min.h5", "rb") as image_file:
        model_min = base64.b64encode(image_file.read())
    # maxsalary model
    history_max = nn.fit(
        x_train, y_train['maxsalary'], epochs=20, batch_size=100, verbose=2)
    y_pred_test_nn_max = nn.predict(x_test)

    max_MSE = mean_squared_error(y_test['maxsalary'], y_pred_test_nn_max)
    max_RMSE = mean_squared_error(
        y_test['maxsalary'], y_pred_test_nn_max, squared=False)
    max_R2 = r2_score(y_test['maxsalary'], y_pred_test_nn_max)
    max_adj_R2 = 1-(1-r2_score(y_test['maxsalary'], y_pred_test_nn_max))*(
        (x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1))

    print('Max salary MSE: ' + str(max_MSE))
    print('Max salary RMSE: ' + str(max_RMSE))
    print('Max salary R-square:' + str(max_R2))
    print('Max salary Adjusted R2: ' + str(max_adj_R2))

    # Save Model
    nn.save("model_max.h5")
    with open("model_max.h5", "rb") as image_file:
        model_max = base64.b64encode(image_file.read())
    # Save One-Hot-Encoder
    with open("encoder.pickle", "wb") as f:
        pickle.dump(enc, f)
    with open("encoder.pickle", "rb") as image_file:
        encoder = base64.b64encode(image_file.read())
    # Save Count Vectorizer
    with open("count_vectorizer.pickle", "wb") as f:
        pickle.dump(count_vectorizer, f)
    with open("count_vectorizer.pickle", "rb") as image_file:
        countvectorizer = base64.b64encode(image_file.read())

    # Save Result in Model DB
    with db.connect() as conn:
        db_row = conn.execute(
            "select count(*) from model")

        if db_row != 0:
            conn.execute(
                "update model set selected = 0 where id = (select max(id) from model)")

        conn.execute(
            "insert into model values (default, 'NN', now(), " + str(min_RMSE) + ", " + str(min_adj_R2) + ", " + str(min_R2) + ", " + str(max_RMSE) + ", " + str(max_adj_R2) + ", " + str(max_R2) + ", 1,"+str(model_min)[1:]+","+str(model_max)[1:]+","+str(encoder)[1:]+","+str(countvectorizer)[1:]+")")

    # select * from careers where error is not null and fixed = "included"
    # fixed can be null -> yet to fixed, fixed => excluded, fixed => included
    requests.get(
        "https://us-central1-fine-climber-348413.cloudfunctions.net/sendmessage?message=Training%20Ended%20at%20"+str(datetime.now())[0:-7])

    return 0


@app.route("/predict", methods=['POST'])
def predict():
    # model = pickle.load("gs://sadsaas/asd.model")
    # return model.predict(job)
    print(request.get_json())
    # Dummy Data
    # 'skills': "|".join([x['value'].replace(" ", "_") for x in request.get_json()['jobSkills']]),
    data = {
        'categories':  "|".join([x['value'].replace(" ", "_") for x in request.get_json()['jobCategory']])+"|".join([x['value'].replace(" ", "_") for x in request.get_json()['jobType']]),
        'positionlevels':  "|".join([x['value'].replace(" ", "_") for x in request.get_json()['jobPositionLevels']]),
        # 'employmenttypes':  "|".join([x['value'].replace(" ", "_") for x in request.get_json()['jobType']]),
        'minimumyearsexperience': int(request.get_json()['minimumYOE']),
        'numberofvacancies': int(request.get_json()['numberofvacancies']),
    }
    print(data)

    x_test = pd.DataFrame(data, index=[0])

    # with db.connect() as conn:
    #     x_test = pd.read_sql(
    #         "select * from careers where uuid='92608a6f62190f2425c5259206728352'", conn)
    # if not (os.path.exists("encoder.pickle") and os.path.exists("count_vectorizer.pickle") and os.path.exists("model_min.h5") and os.path.exists("model_max.h5")):
    #     return Response(json.dumps({"pMinSal": 0, "pMaxSal": 0}),  mimetype='application/json')
    with db.connect() as conn:
        stats = pd.read_sql(
            "select minmodel,maxmodel,enc,countvectorizer from model where selected = 1", conn)
    try:
        with open("model_min.h5", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 0], encoding='utf-8')))
    except Exception as e:
        print(str(e))
        return Response(json.dumps({"pMinSal": 0, "pMaxSal": 0}),  mimetype='application/json')
    try:
        with open("model_max.h5", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 1], encoding='utf-8')))
    except Exception as e:
        print(str(e))
        return Response(json.dumps({"pMinSal": 0, "pMaxSal": 0}),  mimetype='application/json')
    try:
        with open("encoder.pickle", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 2], encoding='utf-8')))
    except Exception as e:
        print(str(e))
        return Response(json.dumps({"pMinSal": 0, "pMaxSal": 0}),  mimetype='application/json')

    try:
        with open("count_vectorizer.pickle", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 3], encoding='utf-8')))
    except Exception as e:
        print(str(e))
        return Response(json.dumps({"pMinSal": 0, "pMaxSal": 0}),  mimetype='application/json')

    model_min = keras.models.load_model("model_min.h5")
    model_max = keras.models.load_model("model_max.h5")

    # Load one hot encoder
    f_enc = open("encoder.pickle", "rb")
    enc = pickle.load(f_enc)

    f_vect = open("count_vectorizer.pickle", "rb")
    count_vectorizer = pickle.load(f_vect)

    # Word Vectorizer
    x_test_categories = count_vectorizer.transform(x_test["categories"])
    x_test_categories_df = pd.DataFrame(
        x_test_categories.todense(), columns=count_vectorizer.get_feature_names())

    # This are column that are categorical
    categorical = ['positionlevels']
    x_test_one_hot_data = enc.transform(x_test[categorical]).toarray()

    feature_name = enc.get_feature_names()
    x_test_one_hot_data_df = pd.DataFrame(
        x_test_one_hot_data, columns=feature_name)

    x_test = pd.concat([x_test_categories_df, x_test_one_hot_data_df, x_test[[
                       'minimumyearsexperience', 'numberofvacancies']].reset_index(drop=True), ], axis=1)

    y_pred_test_nn_min = model_min.predict(x_test)
    y_pred_test_nn_max = model_max.predict(x_test)

    return Response(json.dumps({"pMinSal": int(y_pred_test_nn_min[0][0]), "pMaxSal": int(y_pred_test_nn_max[0][0])}),  mimetype='application/json')


@app.route("/stats")
def stats():
    # with db.connect() as conn:
    #     x_test = pd.read_sql("select * from model", conn)
    # rsquarevalue = [{"name": str(
    #     x+1)+" Jun 2022",  "Max R² Square Value":  random.randint(80, 100), "Min R² Square Value":  random.randint(80, 100)} for x in range(8)]
    # RMSE = [{"name": str(
    #     x+1)+" Jun 2022",  "Max RMSE":  random.randint(800, 1000)/1000}, "Min RMSE":  random.randint(800, 1000)/1000} for x in range(8)]
    # newjob = [{"name": str(
    #     x+1)+" Jun 2022",  "New Job":  random.randint(140, 200)} for x in range(8)]
    with db.connect() as conn:
        stats = pd.read_sql(
            "select to_char(\"createdDate\", 'DD Mon YY, HH24:MI')  as day, min_rmse, min_rsquare, max_rmse, max_rsquare from model order by 1 desc limit 5", conn)
        rsquarevalue = [{"name": str(x), "Min R² Square Value":  round(y, 3), "Max R² Square Value": round(z, 3)}
                        for x, y, z in zip(stats["day"][::-1], stats["min_rsquare"][::-1], stats["max_rsquare"][::-1])]
        RMSE = [{"name": str(x), "Min RMSE":  round(y, 2), "Max RMSE": round(z, 2)}
                for x, y, z in zip(stats["day"][::-1], stats["min_rmse"][::-1], stats["max_rmse"][::-1])]
        stats = pd.read_sql(
            "select Date(crawldate) as day, count(*) as count from careers group by DATE(crawldate) order by 1 desc limit 5;", conn)
        newjob = [{"name": str(x), "New Job":  round(y, 2)}
                  for x, y in zip(stats["day"][::-1], stats["count"][::-1])]
    return Response(json.dumps({"rsquarevalue": rsquarevalue, "RMSE": RMSE, "newjob": newjob}), 200, mimetype='application/json')


@ app.route("/outlier")
def data():
    df = pd.read_sql(
        "select uuid, title, left(description,50) as description, skills, numberofvacancies, categories, positionlevels, postedcompany,employmenttypes, minsalary, maxsalary , remarks from careers where status = 1", db.connect())

    return Response(json.dumps([{v: x[k] for (k, v) in enumerate(df)}for x in df.values]
                               ), 200,  mimetype='application/json')


@ app.route('/outlier', methods=['PUT'])
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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)

# ASCII ART FROM: https://patorjk.com/software/taag/#p=display&f=Small&t=HALF%20CRAWL
