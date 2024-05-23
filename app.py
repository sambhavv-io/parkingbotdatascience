import openai
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings, ChatCohere
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
import re
from datetime import datetime, timedelta

openai.api_version = "2024-02-01"
openai.base_url = "https://parkingarmsai.openai.azure.com/"
COHERE_API_KEY = "hEVI0ZaThsORXlHFEA4f7hEvlyQJMuiXPjt3s1V3"
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")
book_docsearch_location = FAISS.load_local('parking_data/locations', embeddings, allow_dangerous_deserialization=True)
book_docsearch_bookings = FAISS.load_local('parking_data/bookings', embeddings, allow_dangerous_deserialization=True)
book_docsearch_transactions = FAISS.load_local('parking_data/transcations', embeddings,
                                               allow_dangerous_deserialization=True)

app = Flask(__name__)
CORS(app)
llm = ChatCohere(temperature=0, cohere_api_key=COHERE_API_KEY, model="command-r")
# llm = AzureChatOpenAI(azure_deployment="parking_aims", openai_api_version="2024-02-01",
#                       api_key="be1d1118b69349eeb217e369984dfd48"
#                       , azure_endpoint="https://parkingarmsai.openai.azure.com/")

deployment_name = "parking_arms_instruct"

client = AzureOpenAI(
    api_key="be1d1118b69349eeb217e369984dfd48",
    api_version="2024-02-01",
    azure_endpoint="https://parkingarmsai.openai.azure.com/"
)


def find_topic(query):
    topics = ['transcations', 'locations', 'bookings', 'general']
    prompt = f"Classify the following question into one of these topics: '{','.join(topics)}': '{query}'"
    response = client.completions.create(model=deployment_name, prompt=prompt, max_tokens=10)
    return response.choices[0].text.strip().lower()


def find_intent(user_chat):
    prompt = f"""Classify the following message as if user wants to book a slot or query. Also return the probability of it being spam.
    Message: '{user_chat}'.
    The output should only contain one word: booking or query.
    """
    response = client.completions.create(model=deployment_name, prompt=prompt, max_tokens=10)
    return response.choices[0].text.strip()


def extract_time_from_string(input_string):
    # Patterns for different time expressions
    patterns = [
        (re.compile(r'in (\d+(\.\d+)?) hours?'), 'hours'),  # "in 2 hours" or "in 2.5 hours"
        (re.compile(r'in (\d+) hours? (\d+) minutes?'), 'hours_minutes'),  # "in 2 hours 40 minutes"
        (re.compile(r'tomorrow around (\d+)(am|pm)'), 'tomorrow'),  # "tomorrow around 2pm"
        (re.compile(r'after (\d+) days?'), 'days'),  # "after 7 days"
        (re.compile(r'next month'), 'next_month'),  # "next month"
        (re.compile(r'next (\w+)'), 'next_weekday')  # "next Tuesday"
    ]

    for pattern, pattern_type in patterns:
        match = pattern.search(input_string)
        if match:
            if pattern_type == 'hours':
                hours = float(match.group(1))
                return timedelta(hours=hours)
            elif pattern_type == 'hours_minutes':
                hours = int(match.group(1))
                minutes = int(match.group(2))
                return timedelta(hours=hours, minutes=minutes)
            elif pattern_type == 'tomorrow':
                hour = int(match.group(1))
                period = match.group(2)
                if period == 'pm' and hour != 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
                return 'tomorrow', hour
            elif pattern_type == 'days':
                days = int(match.group(1))
                return timedelta(days=days)
            elif pattern_type == 'next_month':
                return 'next_month'
            elif pattern_type == 'next_weekday':
                weekday = match.group(1).lower()
                return 'next_weekday', weekday
    return None


def calculate_future_time(time_info):
    current_time = datetime.now()

    if isinstance(time_info, tuple) and time_info[0] == 'tomorrow':
        hour = time_info[1]
        tomorrow = current_time + timedelta(days=1)
        future_time = datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour)
    elif isinstance(time_info, tuple) and time_info[0] == 'next_weekday':
        target_weekday = time_info[1]
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        target_index = weekdays.index(target_weekday)
        current_weekday_index = current_time.weekday()
        days_until_target = (target_index - current_weekday_index + 7) % 7
        if days_until_target == 0:
            days_until_target = 7  # If today is the target weekday, move to the next occurrence
        future_time = current_time + timedelta(days=days_until_target)
    elif time_info == 'next_month':
        next_month = current_time.month % 12 + 1
        year = current_time.year + (current_time.month // 12)
        day = min(current_time.day, 28)  # To avoid issues with February
        future_time = datetime(year, next_month, day, current_time.hour, current_time.minute, current_time.second)
    elif isinstance(time_info, timedelta):
        future_time = current_time + time_info
    else:
        future_time = current_time  # Default case if no match found (should not happen)

    return future_time


@app.route("/chatgpt", methods=['GET', 'POST'])
def chat_return():
    try:
        user_chat = request.get_json()['user_chat'].lower()
        intent = find_intent(user_chat)
        prompt = user_chat
        chunks_to_retrieve = 5
        if intent == "booking":
            message = "Ofcourse i can help you book a slot please confirm to book a slot?"
            return jsonify({"output": message, "intent": intent})
        else:
            the_topic = find_topic(user_chat)
            if the_topic == "locations":
                retriever_locations = book_docsearch_location.as_retriever(search_type="similarity",
                                                                           search_kwargs={"k": chunks_to_retrieve})
                qa = RetrievalQA.from_llm(llm=llm, retriever=retriever_locations, verbose=True)
                res = qa({"query": prompt})["result"]
                return jsonify({"output": res, "intent": intent})
            elif the_topic == "transactions":
                retriever_transcations = book_docsearch_transactions.as_retriever(search_type="similarity",
                                                                                  search_kwargs={
                                                                                      "k": chunks_to_retrieve})
                qa = RetrievalQA.from_llm(llm=llm, retriever=retriever_transcations, verbose=True)
                res = qa({"query": prompt})["result"]
                return jsonify({"output": res, "intent": intent})
            elif the_topic == "bookings":
                retriever_bookings = book_docsearch_bookings.as_retriever(search_type="similarity",
                                                                          search_kwargs={"k": chunks_to_retrieve})
                qa = RetrievalQA.from_llm(llm=llm, retriever=retriever_bookings, verbose=True)
                res = qa({"query": prompt})["result"]
                return jsonify({"output": res, "intent": intent})
            else:
                return jsonify({"output": "Sorry As an ParkingAgent i cant answer out of context.", "intent": intent})
    except Exception as e:
        return jsonify(repr(e))


@app.route("/timefromstring")
def future_time():
    input_string = request.get_json()['user_chat'].lower()
    hours = extract_time_from_string(input_string)
    if hours is not None:
        future_time = calculate_future_time(hours)
        return jsonify({"output": future_time.strftime('%Y-%m-%d %H:%M:%S')})
    else:
        return jsonify({"output": "No time information found in the input string."})


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")

# print(find_topic("Please share the last 5 invoices?"))
