import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from datetime import datetime
import logging

logging.getLogger("google_genai").setLevel(logging.ERROR)

load_dotenv()
client = genai.Client(http_options={"timeout": 30000})

MODEL = "gemini-3.5-flash"
HISTORY_FILE = Path("history.json")
FACTS_FILE = Path("facts.txt")
NOTES_FILE = Path("notes.txt")


def get_current_time() -> str:
    """Returns the current date and time. Use when the user asks for the time or date."""
    print("  [tool used: get_current_time]")
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def save_note(note: str) -> str:
    """Saves a note for the user. Use when the user asks to note down or save something."""
    print("  [tool used: save_note]")
    with NOTES_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{datetime.now():%d %b %Y %H:%M}] {note}\n")
    return "Note saved."


def read_notes() -> str:
    """Returns all of the user's saved notes. Use when the user asks what notes they have."""
    print("  [tool used: read_notes]")
    if not NOTES_FILE.exists():
        return "There are no notes yet."
    return NOTES_FILE.read_text(encoding="utf-8")

SYSTEM_PROMPT = (
    "You are JARVIS, a witty, polite and highly capable personal AI assistant "
    "for a college student. Address the user as 'sir' or 'Gaurav'. "
    "Keep replies short and sharp, with a touch of dry humor. "
    "Be proactive: suggest next steps when useful. "
    "If you don't know something, say so honestly instead of guessing."
    "You have tools to check the current time, save notes and read notes: use them when relevant. You still cannot browse the web or check live weather, so never claim to."
)

def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return []


def save_history(history):
    HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_system_prompt():
    facts = FACTS_FILE.read_text(encoding="utf-8") if FACTS_FILE.exists() else ""
    if facts.strip():
        return SYSTEM_PROMPT + "\n\nFacts about the user (always keep these in mind):\n" + facts
    return SYSTEM_PROMPT


def ask(history):
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
        for m in history
    ]
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            tools=[get_current_time, save_note, read_notes],
        ),
    )
    return response.text


history = load_history()
print(f"Assistant ready. {len(history) // 2} past exchanges remembered.")
print("Type '/remember <fact>' to save a fact, or 'quit' to exit.")

while True:
    user_message = input("You: ")
    if user_message.lower() in ("quit", "exit"):
        print("Bye!")
        break
    if user_message.startswith("/remember "):
        fact = user_message[len("/remember "):].strip()
        with FACTS_FILE.open("a", encoding="utf-8") as f:
            f.write("- " + fact + "\n")
        print("Assistant: Noted and saved.")
        continue
    history.append({"role": "user", "text": user_message})

    try:
        reply = ask(history)
    except Exception as e:
        print("Assistant: Sorry, something went wrong:", e)
        history.pop()
        continue
    history.append({"role": "model", "text": reply})
    save_history(history)
    print("Assistant:", reply)