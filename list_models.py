from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(http_options={"timeout": 30000})

for m in client.models.list():
    print(m.name)