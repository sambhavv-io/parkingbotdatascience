import requests

api_url = "http://127.0.0.1:5000/chatgpt"

r = requests.post(url=api_url, json={"user_chat": "Please share all bookings ids"})
print(r.text)
