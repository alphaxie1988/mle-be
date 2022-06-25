import os
import telegram
import requests
import uuid


bot = telegram.Bot(token=os.environ["TELEGRAM_TOKEN"])
print("KEY", os.environ["TELEGRAM_TOKEN"])


def webhook(request):
    if request.method == "POST":
        try:
            update = telegram.Update.de_json(request.get_json(force=True), bot)
            chat_id = update.message.chat.id
            if str(update.message.text) == "/crawl":
                bot.sendMessage(
                    chat_id=chat_id, text="Noted, will start crawling if crawling have not started")
                requests.get("https://mle-be-zolecwvnzq-uc.a.run.app/crawl")
            elif str(update.message.text) == "/resetcrawl":
                # Reply with the same message
                reply = requests.get(
                    "https://mle-be-zolecwvnzq-uc.a.run.app/resetcrawl")
                bot.sendMessage(chat_id=chat_id, text=str(reply))
            elif str(update.message.text) == "/plot":
                # Reply with the same message
                bot.sendMessage(
                    chat_id=chat_id, text="https://mle-be-zolecwvnzq-uc.a.run.app/minplot.png?time="+str(uuid.uuid4()))
                bot.sendMessage(
                    chat_id=chat_id, text="https://mle-be-zolecwvnzq-uc.a.run.app/maxplot.png?time="+str(uuid.uuid4()))
            else:
                bot.sendMessage(
                    chat_id=chat_id, text="I do not understand you, you can try to use /crawl to start crawling, /resetcrawl to reset the crawling, /plot to see the error plot.")
            return "ok"
        except:
            return "not ok"
    return "ok"
