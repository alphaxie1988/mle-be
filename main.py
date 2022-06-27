#   ___ __  __ ___  ___  ___ _____
#  |_ _|  \/  | _ \/ _ \| _ \_   _|
#   | || |\/| |  _/ (_) |   / | |
#  |___|_|  |_|_|  \___/|_|_\ |_|

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io
import datetime
import logging
import os
import requests
from flask import Flask, request, Response
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
import pickle
import base64
from xgboost import XGBRegressor

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


try:
    db = init_db_connection()
except:
    print("fail to connect to database")
#   ___ ___ ___ _____ ___ _   _ _        _   ___ ___
#  | _ \ __/ __|_   _| __| | | | |      /_\ | _ \_ _|
#  |   / _|\__ \ | | | _|| |_| | |__   / _ \|  _/| |
#  |_|_\___|___/ |_| |_|  \___/|____| /_/ \_\_| |___|


@app.route('/', methods=['GET'])
def index():
    return Response(
        status=200,
        response="Up and Running: Student Project by SMU"
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
        # Start to Crawl Code Here
    print("Start to Crawl")
    with db.connect() as conn:
        conn.execute("UPDATE mle SET value = '1' WHERE key='crawling'")
    id = str(datetime.now())[14:-7]
    requests.get(
        "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__"+str(datetime.now())[0:-7]+"__MLP__%0AID:%20"+id+"%0A1.%20*🏁Job%20Started*%0A2.%20🐛Crawl%20Started")

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
        requests.get(
            "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__"+str(datetime.now())[0:-7]+"__MLP__%0AID:%20"+id+"%0ACRAWL_ERROR")
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
    clean(id)
    # 4) update database to say crawling ended
    with db.connect() as conn:
        conn.execute("UPDATE mle SET value = '0' WHERE key='crawling'")
    print("Crawl Finish")

    # 5) Count number of new job
    with db.connect() as conn:
        numberOfJobAfter = int(conn.execute(
            "SELECT count(*) FROM careers").fetchone()[0])
    requests.get(
        "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__"+str(datetime.now())[0:-7]+"__MLP__%0AID:%20"+id+"%0A10.%20*🏁Job%20Ended*%0A🆕Number%20Of%20New%20Jobs:%20"+str(numberOfJobAfter-numberOfJobBefore))
    # End of Crawl
    return f"Thank you for waiting!"
    # return Response(
    # status=200,
    # response="Thank you for waiting!"
    # )


#    ___ _    ___   _   _  _ ___ _  _  ___
#   / __| |  | __| /_\ | \| |_ _| \| |/ __|
#  | (__| |__| _| / _ \| .` || || .` | (_ |
#   \___|____|___/_/ \_\_|\_|___|_|\_|\___|

def clean(id):
    requests.get(
        "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__" +
        str(datetime.now())[0:-7]+"__MLP__%0AID:%20"+id+"%0A3.%20🐛Crawl%20Ended%0A4.%20🧹Cleaning%20Started")
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
            "update careers set status = 1, remarks=Concat(remarks,'Max salary is too high|') where maxsalary >= 30000 and status = 0")
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
        "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__"+str(datetime.now())[0:-7]+"__MLP__%0AID:%20"+id+"%0A5.%20🧹Cleaning%20Ended%0A🚩Number%20Of%20Flagged:%20"+str(numberOfFlagedAfter-numberOfFlagedBefore)+"%0A6.%20🏋Training%20Started")
    train(id)

#   _____ ___    _   ___ _  _ ___ _  _  ___
#  |_   _| _ \  /_\ |_ _| \| |_ _| \| |/ __|
#    | | |   / / _ \ | || .` || || .` | (_ |
#    |_| |_|_\/_/ \_\___|_|\_|___|_|\_|\___|


def train(id):

    with db.connect() as conn:
        train_days = pd.read_sql(
            "select value from mle where key = 'numOfDaysToTrain'", conn).iloc[0, 0]
        # Select data for training
        df_raw = pd.read_sql("select * from careers where originalpostingdate >= current_date - INTERVAL '" +
                             str(train_days) + " day' and status in (2,3)", conn)

    df_main = df_raw.copy()
    df_main = df_main[df_main["minimumyearsexperience"] <= 25]
    df_main.fillna(value='', inplace=True)
    # prepare column categories and employment for word vectorizer
    df_main['categories'] = df_main['categories'].str.replace(' / ', '_')
    df_main['categories'] = df_main['categories'].str.replace(' ', '_')
    df_main['categories'] = df_main['categories'].str.replace('|', ' ')
    df_main['employmenttypes'] = df_main['employmenttypes'].str.replace(
        ' / ', '_')
    df_main['employmenttypes'] = df_main['employmenttypes'].str.replace(
        ' ', '_')
    df_main['employmenttypes'] = df_main['employmenttypes'].str.replace(
        '|', ' ')
    # create a 'new col that is categories + employmenttype
    df_main['new_col'] = df_main[['categories', 'employmenttypes']].apply(
        lambda x: '|'.join(x.dropna().values.tolist()), axis=1)

    # Train Test Split
    x_train, x_test, y_train, y_test = model_selection.train_test_split(df_main.loc[:, ~(df_main.columns.isin(
        ['avgsalary', 'minsalary', 'maxsalary']))], df_main[["minsalary", "maxsalary"]], test_size=0.2, random_state=2021)

    # Word vectorizer
    # min_df = 0.01, max_df = 0.5, stop_words = 'english'
    count_vectorizer = feature_extraction.text.CountVectorizer()
    # fit dont put fit into test - fit mean you want to fix the module
    x_train_categories_and_type = count_vectorizer.fit_transform(
        x_train["new_col"])
    x_test_categories_and_type = count_vectorizer.transform(x_test["new_col"])
    x_test_categories_and_type_df = pd.DataFrame(
        x_test_categories_and_type.todense(), columns=count_vectorizer.get_feature_names())
    x_train_categories_and_type_df = pd.DataFrame(
        x_train_categories_and_type.todense(), columns=count_vectorizer.get_feature_names())

    x_train = pd.concat([x_train_categories_and_type_df, x_train[['minimumyearsexperience',
                                                                  'numberofvacancies', 'positionlevels']].reset_index(drop=True), ], axis=1)
    x_test = pd.concat([x_test_categories_and_type_df, x_test[['minimumyearsexperience', 'numberofvacancies',
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
    x_train = pd.concat([x_train_categories_and_type_df, x_train_one_hot_data_df, x_train[[
        'minimumyearsexperience', 'numberofvacancies']].reset_index(drop=True), ], axis=1)
    x_test = pd.concat([x_test_categories_and_type_df, x_test_one_hot_data_df, x_test[[
        'minimumyearsexperience', 'numberofvacancies']].reset_index(drop=True), ], axis=1)

    print(x_train.shape)
    #   ___  ___ ___ ___ _  _ ___   __  __  ___  ___  ___ _
    #  |   \| __| __|_ _| \| | __| |  \/  |/ _ \|   \| __| |
    #  | |) | _|| _| | || .` | _|  | |\/| | (_) | |) | _|| |__
    #  |___/|___|_| |___|_|\_|___| |_|  |_|\___/|___/|___|____|

############ Neural Network (Model 1) #################
    # nn = Sequential()
    # nn.add(Dense(78, input_dim=x_train.shape[1], activation='relu'))
    # nn.add(Dropout(0.2))
    # nn.add(Dense(39, activation='relu'))
    # nn.add(Dropout(0.2))
    # nn.add(Dense(19, activation='relu'))
    # nn.add(Dropout(0.2))
    # nn.add(Dense(10, activation='relu'))
    # nn.add(Dropout(0.2))
    # nn.add(Dense(1, kernel_initializer='normal'))
    # # Compile model
    # nn.compile(loss='mean_squared_error', optimizer='adam')
    # # nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # print(x_test.shape)

    #   __  __ ___ _  _   ___   _   _      _   _____   __  __  __  ___  ___  ___ _
    #  |  \/  |_ _| \| | / __| /_\ | |    /_\ | _ \ \ / / |  \/  |/ _ \|   \| __| |
    #  | |\/| || || .` | \__ \/ _ \| |__ / _ \|   /\ V /  | |\/| | (_) | |) | _|| |__
    #  |_|  |_|___|_|\_| |___/_/ \_\____/_/ \_\_|_\ |_|   |_|  |_|\___/|___/|___|____|


############ Neural Network (Model 1) #################
    # nn.fit(x_train, y_train['minsalary'], epochs=20, batch_size=100, verbose=2)
    # y_pred_test_nn_min = nn.predict(x_test)
    # # Evaluation
    # min_MSE = mean_squared_error(y_test['minsalary'], y_pred_test_nn_min)
    # min_RMSE = mean_squared_error(
    #     y_test['minsalary'], y_pred_test_nn_min, squared=False)
    # min_R2 = r2_score(y_test['minsalary'], y_pred_test_nn_min)
    # min_adj_R2 = 1-(1-r2_score(y_test['minsalary'], y_pred_test_nn_min))*(
    #     (x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1))

    # print('Min salary MSE: ' + str(min_MSE))
    # print('Min salary RMSE: ' + str(min_RMSE))
    # print('Min salary R2:' + str(min_R2))
    # print('Min salary Adjusted R2: ' + str(min_adj_R2))


############ XGBoost (Model 2) #################
    xg_reg = XGBRegressor(objective='reg:squarederror',
                          seed=123, n_estimators=10)
    xg_reg.fit(x_train, y_train['minsalary'])
    y_preds_min = xg_reg.predict(x_test)

    min_MSE = mean_squared_error(y_test['minsalary'], y_preds_min)
    min_RMSE = mean_squared_error(
        y_test['minsalary'], y_preds_min, squared=False)
    min_R2 = r2_score(y_test['minsalary'], y_preds_min)
    min_adj_R2 = 1-(1-r2_score(y_test['minsalary'], y_preds_min))*(
        (x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1))

    print('Min salary MSE: ' + str(min_MSE))
    print('Min salary RMSE: ' + str(min_RMSE))
    print('Min salary R2:' + str(min_R2))
    print('Min salary Adjusted R2: ' + str(min_adj_R2))

    feature_important = xg_reg.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    imptfeature = pd.DataFrame(data=values, index=keys, columns=[
                               "score"]).sort_values(by="score", ascending=False)

    top5features_min = list(imptfeature.index)[:10]
    top5features_str_min = ", ".join(top5features_min)

    # Save Model
############ Neural Network (Model 1) #################
    # nn.save("model_min.h5")

############ XGBoost (Model 2) #################
    xg_reg.save_model("model_min.h5")

    with open("model_min.h5", "rb") as image_file:
        model_min = base64.b64encode(image_file.read())

        #   __  __   _   __  __  ___   _   _      _   _____   __  __  __  ___  ___  ___ _
        #  |  \/  | /_\  \ \/ / / __| /_\ | |    /_\ | _ \ \ / / |  \/  |/ _ \|   \| __| |
        #  | |\/| |/ _ \  >  <  \__ \/ _ \| |__ / _ \|   /\ V /  | |\/| | (_) | |) | _|| |__
        #  |_|  |_/_/ \_\/_/\_\ |___/_/ \_\____/_/ \_\_|_\ |_|   |_|  |_|\___/|___/|___|____|

    ############ Neural Network (Model 1) #################
    # nn.fit(
    #     x_train, y_train['maxsalary'], epochs=20, batch_size=100, verbose=2)
    # y_pred_test_nn_max = nn.predict(x_test)

    # max_MSE = mean_squared_error(y_test['maxsalary'], y_pred_test_nn_max)
    # max_RMSE = mean_squared_error(
    #     y_test['maxsalary'], y_pred_test_nn_max, squared=False)
    # max_R2 = r2_score(y_test['maxsalary'], y_pred_test_nn_max)
    # max_adj_R2 = 1-(1-r2_score(y_test['maxsalary'], y_pred_test_nn_max))*(
    #     (x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1))

    # print('Max salary MSE: ' + str(max_MSE))
    # print('Max salary RMSE: ' + str(max_RMSE))
    # print('Max salary R-square:' + str(max_R2))
    # print('Max salary Adjusted R2: ' + str(max_adj_R2))

    ############ XGBoost (Model 2) #################
    xg_reg = XGBRegressor(objective='reg:squarederror',
                          seed=123, n_estimators=10)
    xg_reg.fit(x_train, y_train['maxsalary'])
    y_preds_max = xg_reg.predict(x_test)

    max_MSE = mean_squared_error(y_test['maxsalary'], y_preds_max)
    max_RMSE = mean_squared_error(
        y_test['maxsalary'], y_preds_max, squared=False)
    max_R2 = r2_score(y_test['maxsalary'], y_preds_max)
    max_adj_R2 = 1-(1-r2_score(y_test['maxsalary'], y_preds_max))*(
        (x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1))

    print('Max salary MSE: ' + str(max_MSE))
    print('Max salary RMSE: ' + str(max_RMSE))
    print('Max salary R2:' + str(max_R2))
    print('Max salary Adjusted R2: ' + str(max_adj_R2))

    feature_important = xg_reg.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    imptfeature = pd.DataFrame(data=values, index=keys, columns=[
                               "score"]).sort_values(by="score", ascending=False)

    top5features_max = list(imptfeature.index)[:10]
    top5features_str_max = ", ".join(top5features_max)

    # axis.plot(xs, ys)
    ############ Neural Network (Model 1) #################
    # axis.scatter(y_test['minsalary'], y_pred_test_nn_min,
    #              color='red', alpha=0.1, s=10)
    ############ XGBoost (Model 2) #################

    # for image R2
    plt.figure(1)
    plt.title('Error of Minimum Salary')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.scatter(y_test['minsalary'], y_preds_min,
                color='red', alpha=0.1, s=10)
    output = io.BytesIO()
    FigureCanvas(plt.gcf()).print_png(output)
    minb64 = str(base64.b64encode(output.getvalue()))

    # axis.plot(xs, ys)
    ############ Neural Network (Model 1) #################
    # axis2.scatter(y_test['maxsalary'], y_pred_test_nn_max,
    #               color='green', alpha=0.1, s=10)
    ############ XGBoost (Model 2) #################
    plt.figure(2)
    plt.title('Error of Maximum Salary')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.scatter(y_test['maxsalary'], y_preds_max,
                color='green', alpha=0.1, s=10)
    output2 = io.BytesIO()
    FigureCanvas(plt.gcf()).print_png(output2)
    maxb64 = str(base64.b64encode(output2.getvalue()))

    # for image R2
    # plt.rcParams["figure.figsize"] = (20, 20)
    # plot_importance(xg_reg, max_num_features=10)

    #   ___   ___   _____   __  __  ___  ___  ___ _
    #  / __| /_\ \ / / __| |  \/  |/ _ \|   \| __| |
    #  \__ \/ _ \ V /| _|  | |\/| | (_) | |) | _|| |__
    #  |___/_/ \_\_/ |___| |_|  |_|\___/|___/|___|____|

    ############ Neural Network (Model 1) #################
    # nn.save("model_max.h5")

    ############ XGBoost (Model 2) #################
    xg_reg.save_model("model_max.h5")

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
                "update model set selected = 0")

        conn.execute(
            "insert into model values (default, 'XGBoost', now(), " + str(min_RMSE) + ", " + str(min_adj_R2) + ", " + str(min_R2) + ", " + str(max_RMSE) + ", " + str(max_adj_R2) + ", " + str(max_R2) + ", 1,"+str(model_min)[1:]+","+str(model_max)[1:]+","+str(encoder)[1:]+","+str(countvectorizer)[1:]+","+minb64[1:]+","+maxb64[1:]+",'"+top5features_str_min+"','"+top5features_str_max+"')")

    # select * from careers where error is not null and fixed = "included"
    # fixed can be null -> yet to fixed, fixed => excluded, fixed => included
    requests.get(
        "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__"+str(datetime.now())[0:-7]+"__MLP__%0AID:%20"+id+"%0A7.%20🏋Training%20Ended%0AMin_RMSE:%20" + str(round(min_RMSE, 3))+"%0AMax_RMSE:%20" + str(round(max_RMSE, 3))+"%0AMin_R2:%20" + str(round(min_R2, 3))+"%0AMax_R2:%20" + str(round(max_R2, 3))+"%0A8.%20🧪Testing%20Started")
    testingEndPoint(id)


def testingEndPoint(id):
    url = 'https://mle-be-zolecwvnzq-uc.a.run.app/predict'
    myobj = {"numberofvacancies": 1, "jobCategory": [],
             "jobType": [], "jobPositionLevels": [], "minimumYOE": "1"}
    try:
        result = json.loads(
            requests.post(url, json=myobj).content)
        if(result["pMinSal"] > 0 and result["pMaxSal"] > 0):
            requests.get(
                "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__"+str(datetime.now())[0:-7]+"__MLP__%0AID:%20"+id+"%0A9.%20🧪Testing%20Ended%0AResult:%20OK✅")
    except:
        requests.get(
            "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__"+str(datetime.now())[0:-7]+"__MLP__%0AID:%20"+id+"%0A9.%20🧪Testing%20Ended%0AResult:%20FAIL❌")


#   ___ ___ ___ ___ ___ ___ _____
#  | _ \ _ \ __|   \_ _/ __|_   _|
#  |  _/   / _|| |) | | (__  | |
#  |_| |_|_\___|___/___\___| |_|


@ app.route("/predict", methods=['POST'])
def predict():
    data = {
        'categories':  "|".join([x['value'].replace(" ", "_") for x in request.get_json()['jobCategory']])+"|".join([x['value'].replace(" ", "_") for x in request.get_json()['jobType']]),
        'positionlevels':  "|".join([x['value'].replace(" ", "_") for x in request.get_json()['jobPositionLevels']]),
        'minimumyearsexperience': int(request.get_json()['minimumYOE']),
        'numberofvacancies': int(request.get_json()['numberofvacancies']),
    }
    print(data)

    x_test = pd.DataFrame(data, index=[0])

    if not (os.path.exists("encoder.pickle") and os.path.exists("count_vectorizer.pickle") and os.path.exists("model_min.h5") and os.path.exists("model_max.h5")):
        return Response(json.dumps({"pMinSal": 0, "pMaxSal": 0}),  mimetype='application/json')

    # Word Vectorizer
    x_test_categories_and_type = count_vectorizer.transform(
        x_test["categories"])
    x_test_categories_and_type_df = pd.DataFrame(
        x_test_categories_and_type.todense(), columns=count_vectorizer.get_feature_names())

    # This are column that are categorical
    categorical = ['positionlevels']
    x_test_one_hot_data = enc.transform(x_test[categorical]).toarray()

    feature_name = enc.get_feature_names()
    x_test_one_hot_data_df = pd.DataFrame(
        x_test_one_hot_data, columns=feature_name)

    x_test = pd.concat([x_test_categories_and_type_df, x_test_one_hot_data_df, x_test[[
        'minimumyearsexperience', 'numberofvacancies']].reset_index(drop=True), ], axis=1)

    y_pred_test_nn_min = model_min.predict(x_test)
    y_pred_test_nn_max = model_max.predict(x_test)

    return Response(json.dumps({"pMinSal": int(y_pred_test_nn_min[0]), "pMaxSal": int(y_pred_test_nn_max[0])}), 200, mimetype='application/json')

#   ___ _____ _ _____ ___ ___ _____ ___ ___
#  / __|_   _/_\_   _|_ _/ __|_   _|_ _/ __|
#  \__ \ | |/ _ \| |  | |\__ \ | |  | | (__
#  |___/ |_/_/ \_\_| |___|___/ |_| |___\___|


@ app.route("/stats")
def stats():
    with db.connect() as conn:
        stats = pd.read_sql(
            "select to_char(\"createdDate\", 'DD Mon YY, HH24:MI')  as day, min_rmse, min_rsquare, max_rmse, max_rsquare from model order by 1 desc limit 7", conn)
        rsquarevalue = [{"name": str(x), "Min R² Value":  round(y, 3), "Max R² Value": round(z, 3)}
                        for x, y, z in zip(stats["day"][::-1], stats["min_rsquare"][::-1], stats["max_rsquare"][::-1])]
        RMSE = [{"name": str(x), "Min RMSE":  round(y, 2), "Max RMSE": round(z, 2)}
                for x, y, z in zip(stats["day"][::-1], stats["min_rmse"][::-1], stats["max_rmse"][::-1])]
        stats = pd.read_sql(
            "select Date(crawldate) as day, count(*) as count from careers group by DATE(crawldate) order by 1 desc limit 7;", conn)
        newjob = [{"name": str(x), "New Job":  round(y, 2)}
                  for x, y in zip(stats["day"][::-1], stats["count"][::-1])]
    return Response(json.dumps({"rsquarevalue": rsquarevalue, "RMSE": RMSE, "newjob": newjob}), 200, mimetype='application/json')


#    ___  _   _ _____ _    ___ ___ ___
#   / _ \| | | |_   _| |  |_ _| __| _ \
#  | (_) | |_| | | | | |__ | || _||   /
#   \___/ \___/  |_| |____|___|___|_|_\

@ app.route("/outlier")
def data():
    df = pd.read_sql(
        "select uuid, title, left(description,50) as description, skills, numberofvacancies, categories, positionlevels, postedcompany,employmenttypes, minsalary, maxsalary , remarks, minimumyearsexperience from careers where status = 1", db.connect())

    return Response(json.dumps([{v: x[k] for (k, v) in enumerate(df)}for x in df.values]
                               ), 200,  mimetype='application/json')


@ app.route('/outlier', methods=['PUT'])
def updateData():
    # print(request.get_json()['action'])
    # print(str(request.get_json()['payload'])[1:-1])
    if request.get_json()['action'] == "Add":
        with db.connect() as conn:
            conn.execute(
                "update careers set status = 3, remarks=Concat(remarks,'Admin marked as OK|') where uuid in ("+str(request.get_json()['payload'])[1:-1]+")")
    elif request.get_json()['action'] == "Hide":
        with db.connect() as conn:
            conn.execute(
                "update careers set status = 4, remarks=Concat(remarks,'Admin marked as not OK|') where uuid in ("+str(request.get_json()['payload'])[1:-1]+")")
    return Response(json.dumps({"success": True}), 200,  mimetype='application/json')

#   __  __  ___  ___  ___ _
#  |  \/  |/ _ \|   \| __| |
#  | |\/| | (_) | |) | _|| |__
#  |_|  |_|\___/|___/|___|____|


@ app.route("/model")
def modellist():
    df = pd.read_sql("select id,\"createdDate\", min_rmse, min_adjrsquare, min_rsquare, max_rmse, max_adjrsquare, max_rsquare, selected,modelfilename, minfeature, maxfeature FROM public.model order by 1 desc limit 7", db.connect())

    return Response(json.dumps([{v: str(x[k]) for (k, v) in enumerate(df)}for x in df.values]
                               ), 200,  mimetype='application/json')


@ app.route('/model', methods=['PUT'])
def chooseModel():
    with db.connect() as conn:
        conn.execute(
            "update model set selected = 0")
        conn.execute(
            "update model set selected = 1 where id in ("+str(request.get_json()['payload'])+")")
    loadModel()
    return Response(json.dumps({"success": True}), 200,  mimetype='application/json')

#   ___ ___ ___ _    ___   _   ___    __  __  ___  ___  ___ _
#  | _ \ _ \ __| |  / _ \ /_\ |   \  |  \/  |/ _ \|   \| __| |
#  |  _/   / _|| |_| (_) / _ \| |) | | |\/| | (_) | |) | _|| |__
#  |_| |_|_\___|____\___/_/ \_\___/  |_|  |_|\___/|___/|___|____|


global model_min
global model_max
global count_vectorizer
global enc


def loadModel():
    global model_min
    global model_max
    global count_vectorizer
    global enc
    with db.connect() as conn:
        stats = pd.read_sql(
            "select minmodel,maxmodel,enc,countvectorizer from model where selected = 1", conn)
    try:
        with open("model_min.h5", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 0], encoding='utf-8')))
    except Exception as e:
        print(str(e))
    try:
        with open("model_max.h5", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 1], encoding='utf-8')))
    except Exception as e:
        print(str(e))
    try:
        with open("encoder.pickle", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 2], encoding='utf-8')))
    except Exception as e:
        print(str(e))

    try:
        with open("count_vectorizer.pickle", "wb") as f:
            f.write(base64.decodebytes(
                bytes(stats.iloc[0, 3], encoding='utf-8')))
    except Exception as e:
        print(str(e))

    try:
        ############## Neural Network (Model1) ###################
        # model_min = keras.models.load_model("model_min.h5")
        # model_max = keras.models.load_model("model_max.h5")

        ############## XGBoost (Model2) ###################
        model_min = XGBRegressor()
        model_min.load_model("model_min.h5")
        model_max = XGBRegressor()
        model_max.load_model("model_max.h5")

        enc = pickle.load(open("encoder.pickle", "rb"))
        count_vectorizer = pickle.load(open("count_vectorizer.pickle", "rb"))
    except Exception as e:
        print(str(e))


try:
    loadModel()
except:
    print("most likely cannot connect to db")

#   ___  ___   _ _____ _____ ___ ___ ___ _    ___ _____
#  / __|/ __| /_\_   _|_   _| __| _ \ _ \ |  / _ \_   _|
#  \__ \ (__ / _ \| |   | | | _||   /  _/ |_| (_) || |
#  |___/\___/_/ \_\_|   |_| |___|_|_\_| |____\___/ |_|


@app.route('/minplot.png')
def plot1_png():
    try:
        b64 = pd.read_sql(
            "select minplot FROM public.model where selected = 1 order by 1 desc limit 1", db.connect()).iloc[0, 0]
        o64 = base64.b64decode(b64)
        return Response(o64, mimetype='image/png')
    except:

        return Response("", mimetype='image/png')


@app.route('/maxplot.png')
def plot2_png():
    try:
        b64 = pd.read_sql(
            "select maxplot FROM public.model where selected = 1 order by 1 desc limit 1", db.connect()).iloc[0, 0]
        o64 = base64.b64decode(b64)
        return Response(o64, mimetype='image/png')
    except:

        return Response("", mimetype='image/png')


#   ___ _____ _   ___ _____     _   ___ ___
#  / __|_   _/_\ | _ \_   _|   /_\ | _ \ _ \
#  \__ \ | |/ _ \|   / | |    / _ \|  _/  _/
#  |___/ |_/_/ \_\_|_\ |_|   /_/ \_\_| |_|

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)


# ASCII ART FROM: https://patorjk.com/software/taag/#p=display&f=Small&t=HALF%20CRAWL
