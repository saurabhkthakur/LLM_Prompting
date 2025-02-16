import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API = os.environ['GOOGLE_API_KEY']

genai.configure(api_key=GOOGLE_API)
flash = genai.GenerativeModel('gemini-1.5-flash')
response = flash.generate_content("Explain AI to me like I'm a kid.")
print(response.text)