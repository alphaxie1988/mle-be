import base64
import json
from datetime import datetime
import requests
import time


def hello_pubsub(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    temp = json.loads(pubsub_message.replace("\'", "\"").replace(
        "True", "\"True\"").replace("False", "\"False\""))
    try:
        if(temp["status"] in ["QUEUED", "SUCCESS", "WORKING", "TIMEOUT", "CANCELLED", "FAILED", "FAILURE"]):
            if(temp["status"] == "SUCCESS"):
                requests.get("https://us-central1-mle-by-xjl.cloudfunctions.net/sendmsg?message=__"+str(datetime.now())[0:-7]+"__CICD__%0ABuild%20Job%20Status%20Updated%0AImage%20:"+str(
                    temp["images"])+"%0AStatus:%20"+str(temp["status"])+"%0Ahttps://rb.gy/cgkz6i")
                time.sleep(20)
                url = 'https://mle-be-zolecwvnzq-uc.a.run.app/predict'
                myobj = {"numberofvacancies": 1, "jobCategory": [],
                         "jobType": [], "jobPositionLevels": [], "minimumYOE": "1"}
                try:
                    result = json.loads(requests.post(url, json=myobj).content)
                    if(result["pMinSal"] > 0 and result["pMaxSal"] > 0):
                        requests.get("https://us-central1-mle-by-xjl.cloudfunctions.net/sendmsg?message=__"+str(datetime.now())[
                                     0:-7]+"__CICD__%0ABuild%20Job%20Status%20Updated%0ATest%20Result%20: OK%0Ahttps://tinyurl.com/2022mle")
                        print("OK")
                except:
                    requests.get("https://us-central1-mle-by-xjl.cloudfunctions.net/sendmsg?message=__"+str(
                        datetime.now())[0:-7]+"__CICD__%0ABuild%20Job%20Status%20Updated%0ATest%20Result%20: FAIL%0Ahttps://rb.gy/nmswrm")
                    print("Fail")
            elif(temp["status"] in ["FAILED", "TIMEOUT", "FAILURE"]):
                requests.get("https://us-central1-mle-by-xjl.cloudfunctions.net/sendmsg?message=__"+str(datetime.now())[0:-7]+"__CICD__%0ABuild%20Job%20Status%20Updated%0AImage%20:"+str(
                    temp["images"])+"%0AStatus:%20"+str(temp["status"])+"%0Ahttps://rb.gy/nmswrm")
            else:
                requests.get("https://us-central1-mle-by-xjl.cloudfunctions.net/sendmsg?message=__"+str(datetime.now())[
                             0:-7]+"__CICD__%0ABuild%20Job%20Status%20Updated%0AImage%20:"+str(temp["images"])+"%0AStatus:%20"+str(temp["status"]))
    except:
        print("Error")
     # print(pubsub_message)
    print("|||"+str(temp)+"|||")
