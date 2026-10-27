from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(http_options={"timeout": 30000})

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Say hello and introduce yourself in one sentence.",
)
print(response.text)
