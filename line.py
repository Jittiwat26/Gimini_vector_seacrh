from flask import Flask, request
from linebot.models import *
from linebot import *
from vertex_ai import promp_ai
import datetime

app = Flask(__name__)


line_bot_api = LineBotApi('I9mgk1gqe7x4tirEd5AjNi5Il91E0iiAF44CeRU/kaQd+NatHpxtBXMNE6PbArDhuIptcyh77FDtueM9f+6gqKAAnVpqxPKKc4DsCNjWuK4IxK8xcxOmzn9j9YE8oavM1wc+o+8Uo8UDgT6uS9XSaQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('514a4a0dbabf4aae1cac76ee28aed860')

@app.route("/healthz", methods=['GET'])
def health_check():
    return "health check"


@app.route("/callback", methods=['POST'])
def callback():
    body = request.get_data(as_text=True)
    print(body)
    req = request.get_json(silent=True, force=True)
    events = req['events']
    if events:
        event = events[0]  # Assuming you want to handle the first event
        reply_token = event['replyToken']
        text = event['message']['text']

        print(f"Reply Token: {reply_token}")
        print(f"Text: {text}")
        system_prompt = (
            "you are Poker assistant"
            "Use the given player_hand to answer determine wether to float or not "
            "player_hand: {context}"
        )
        response = promp_ai(text)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{response}"))
    return 'OK'
def reply(intent,text,reply_token,id,disname, lotto_number, session):
    print(text)
if __name__ == "__main__":
    app.run(host="0.0.0.0")

