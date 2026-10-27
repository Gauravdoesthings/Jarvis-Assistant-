import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(timeout=30)

MODEL = "openai/gpt-oss-120b"
HISTORY_FILE = Path("history.json")
FACTS_FILE = Path("facts.txt")
NOTES_FILE = Path("notes.txt")

SYSTEM_PROMPT = (
    "You are JARVIS, a witty, polite and highly capable personal AI assistant "
    "for a college student. Address the user as 'sir' or 'Gaurav'. "
    "Keep replies short and sharp, with a touch of dry humor. "
    "Be proactive: suggest next steps when useful. "
    "If you don't know something, say so honestly instead of guessing."
    "You currently have no tools: you cannot browse the web, check live weather, set reminders or save notes yet. Never claim or offer to do those things."
)


def get_current_time() -> str:
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def save_note(note: str) -> str:
    with NOTES_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{datetime.now():%d %b %Y %H:%M}] {note}\n")
    return "Note saved."


def read_notes() -> str:
    if not NOTES_FILE.exists():
        return "There are no notes yet."
    return NOTES_FILE.read_text(encoding="utf-8")


AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "save_note": save_note,
    "read_notes": read_notes,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time. Use when the user asks for the time or date.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Saves a note for the user. Use when the user asks to note down or save something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The text of the note to save."}
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": "Returns all of the user's saved notes. Use when the user asks what notes they have.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


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
    messages = [{"role": "system", "content": build_system_prompt()}]
    for m in history:
        role = "assistant" if m["role"] == "model" else "user"
        messages.append({"role": role, "content": m["text"]})

    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            result = AVAILABLE_TOOLS[name](**args)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "name": name, "content": result}
            )
    return "Sorry, I got stuck using my tools."


history = load_history()
print(f"Assistant ready (Groq). {len(history) // 2} past exchanges remembered.")
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