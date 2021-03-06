import base64
import json
from datetime import datetime
import requests
import time
from datetime import datetime, timedelta


def hello_pubsub(event, context):
    singaporeTime = str(datetime.now() + timedelta(hours=8))
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    cloudfunctionMessage = "https://us-central1-mle-by-xjl.cloudfunctions.net/sendMessage?message=__" + \
        singaporeTime[0:-7] + \
        "__CICD__%0ABuild%20Job%20Status%20Updated%0A"
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    temp = json.loads(pubsub_message.replace("\'", "\"").replace(
        "True", "\"True\"").replace("False", "\"False\""))
    try:
        if(temp["status"] in ["QUEUED", "SUCCESS", "WORKING", "TIMEOUT", "CANCELLED", "FAILED", "FAILURE"]):
            if(temp["status"] == "SUCCESS"):
                requests.get(cloudfunctionMessage+"Image%20:"+str(
                    temp["images"])+"%0AStatus:%20"+str(temp["status"])+"%0A😃")
                time.sleep(60)
                if (str(temp["images"]).find("staging") == -1):
                    url = 'https://mle-be-zolecwvnzq-uc.a.run.app/predict'
                    myobj = {"numberofvacancies": 1, "jobCategory": [],
                             "jobType": [], "jobPositionLevels": [], "minimumYOE": "1"}
                    try:
                        result = json.loads(
                            requests.post(url, json=myobj).content)
                        if(result["pMinSal"] > 0 and result["pMaxSal"] > 0):
                            requests.get(
                                cloudfunctionMessage+"Image%20:"+str(
                                    temp["images"])+"%0ATest%20Result%20:%20OK✅%0Ahttps://tinyurl.com/2022mle")
                            print("OK")
                    except:
                        requests.get(
                            cloudfunctionMessage+"Image%20:"+str(
                                temp["images"])+"%0ATest%20Result%20:%20FAIL❌")
                        print("Fail")
                else:
                    url = 'https://mle-be-staging-zolecwvnzq-uc.a.run.app/predict'
                    myobj = {"numberofvacancies": 1, "jobCategory": [],
                             "jobType": [], "jobPositionLevels": [], "minimumYOE": "1"}
                    try:
                        result = json.loads(
                            requests.post(url, json=myobj).content)
                        if(result["pMinSal"] > 0 and result["pMaxSal"] > 0):
                            requests.get(
                                cloudfunctionMessage+"Image%20:"+str(
                                    temp["images"])+"%0ATest%20Result%20:%20OK✅%0Ahttps://tinyurl.com/2022mle-staging")
                            print("OK")
                    except:
                        requests.get(
                            cloudfunctionMessage+"Image%20:"+str(
                                temp["images"])+"%0ATest%20Result%20:%20FAIL❌")
                        print("Fail")

            if(temp["status"] == "TIMEOUT"):
                requests.get(cloudfunctionMessage+"Image%20:" +
                             str(temp["images"])+"%0AStatus:%20"+str(temp["status"])+"%0A⏰")
            if(temp["status"] == "QUEUED"):
                requests.get(cloudfunctionMessage+"Image%20:" +
                             str(temp["images"])+"%0AStatus:%20"+str(temp["status"])+"%0A🚶🚶🚶🚶")
            if(temp["status"] == "WORKING"):
                requests.get(cloudfunctionMessage+"Image%20:" +
                             str(temp["images"])+"%0AStatus:%20"+str(temp["status"])+"%0A🏗️")
            if(temp["status"] == "CANCELLED"):
                requests.get(cloudfunctionMessage+"Image%20:" +
                             str(temp["images"])+"%0AStatus:%20"+str(temp["status"])+"%0A🚫")
            if(temp["status"] in ["FAILED", "FAILURE"]):
                requests.get(cloudfunctionMessage+"Image%20:" +
                             str(temp["images"])+"%0AStatus:%20"+str(temp["status"])+"%0ALog:%20"+str(temp["logUrl"])+"%0A😭")
    except Exception as inst:
        print("Error", inst)
    # print(pubsub_message)
    # print("|||"+str(temp)+"|||")
