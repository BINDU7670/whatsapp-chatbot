from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import pickle
import json
import random
import csv

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

# Save user messages
def save_chat(msg, number):
    with open("data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([number, msg])

# Get response
def get_response(msg):
    X = vectorizer.transform([msg])
    tag = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn’t understand."

# Route
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").lower()
    sender = request.values.get("From", "")

    # Save chat
    save_chat(incoming_msg, sender)

    # Lead collection example
    if "name" in incoming_msg:
        reply = "Please enter your name."
    elif "course" in incoming_msg:
        reply = "Which course are you interested in?"
    else:
        reply = get_response(incoming_msg)

    resp = MessagingResponse()
    resp.message(reply)

    return str(resp)

if __name__ == "__main__":
    app.run(port=5000)