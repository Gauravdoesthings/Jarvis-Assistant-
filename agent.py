import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(timeout=30)

MODEL = "openai/gpt-oss-120b"
HISTORY_FILE = Path("history.json")
FACTS_FILE = Path("facts.txt")
NOTES_FILE = Path("notes.txt")
HABITS_FILE = Path("habits.csv")

SYSTEM_PROMPT = (
    "You are JARVIS, a witty, polite and highly capable personal AI assistant "
    "for a college student. Address the user as 'sir' or 'Gaurav'. "
    "Keep replies short and sharp, with a touch of dry humor. "
    "Be proactive: suggest next steps when useful. "
    "If you don't know something, say so honestly instead of guessing."
    "You currently have no tools: you cannot browse the web, check live weather, set reminders or save notes yet. Never claim or offer to do those things."
    "You can also log habits (workout, spanish, reading, agent) and summarize them. Use exactly those four habit names."
)


# ---------- Tools ----------
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


def log_habit(habit: str, note: str = "") -> str:
    new_file = not HABITS_FILE.exists()
    with HABITS_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["date", "habit", "note"])
        writer.writerow([date.today().isoformat(), habit.lower().strip(), note])
    return f"Logged {habit}."


def habit_summary(days: int = 7) -> str:
    if not HABITS_FILE.exists():
        return "No habits logged yet."
    cutoff = date.today() - timedelta(days=int(days))
    counts = {}
    with HABITS_FILE.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if date.fromisoformat(row["date"]) > cutoff:
                counts[row["habit"]] = counts.get(row["habit"], 0) + 1
    if not counts:
        return f"Nothing logged in the last {days} days."
    return "\n".join(f"{h}: {n} time(s)" for h, n in counts.items())


AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "save_note": save_note,
    "read_notes": read_notes,
    "log_habit": log_habit,
    "habit_summary": habit_summary,
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
    {
        "type": "function",
        "function": {
            "name": "log_habit",
            "description": "Logs that the user completed a habit today. Use when the user says they did a workout, studied Spanish, read their book or worked on their agent project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "habit": {"type": "string", "description": "One of: workout, spanish, reading, agent."},
                    "note": {"type": "string", "description": "Optional short detail, like '20 minutes' or 'chapter 3'."},
                },
                "required": ["habit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "habit_summary",
            "description": "Summarizes how many times each habit was done recently. Use when the user asks about their progress or week.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "How many days back to count. Default 7."}
                },
            },
        },
    },
]


# ---------- Memory ----------
def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return []


def save_history(history):
    HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_facts():
    if FACTS_FILE.exists():
        return FACTS_FILE.read_text(encoding="utf-8")
    return ""


def add_fact(fact):
    with FACTS_FILE.open("a", encoding="utf-8") as f:
        f.write("- " + fact + "\n")


def build_system_prompt():
    facts = read_facts()
    if facts.strip():
        return SYSTEM_PROMPT + "\n\nFacts about the user (always keep these in mind):\n" + facts
    return SYSTEM_PROMPT


# ---------- The brain ----------
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