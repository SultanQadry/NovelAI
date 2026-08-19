from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in .env")

client = genai.Client(api_key=api_key)

chat = client.chats.create(model="gemini-3.6-flash")
response = chat.send_message("Say hello to NovelAI in one short sentence.")

print(response.text)