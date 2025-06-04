import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from note_schema import Note

load_dotenv()

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found.")
        sys.exit(1)
    return OpenAI(api_key=api_key)

def load_assistant_id():
    path = Path(".assistant")
    if not path.exists():
        print("❌ Assistant ID not found. Run 00_bootstrap.py first.")
        sys.exit(1)
    return path.read_text().strip()

def create_summary_prompt():
    return (
        "You are a study summarizer. "
        "Return exactly 10 unique notes that will help prepare for the exam. "
        "Respond ONLY with valid JSON matching this format: "
        "{ \"notes\": [ { \"id\": 1, \"heading\": \"...\", \"summary\": \"...\", \"page_ref\": 3 }, ... ] }. "
        "Each note must include: an integer id (1–10), a short heading, a summary (max 150 characters), and optional page_ref (page number). "
        "Do not include Markdown or explanations. Respond ONLY with raw JSON text."

    )

def create_thread_and_message(client):
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Please summarize the document into 10 study notes."
    )
    return thread.id

def run_assistant(client, assistant_id, thread_id, system_prompt):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=system_prompt,
        response_format={"type": "json_object"}
    )
    while run.status not in ["completed", "failed", "cancelled"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    if run.status != "completed":
        raise RuntimeError(f"Run failed with status: {run.status}")

    return run

def extract_response_text(client, thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for message in reversed(messages.data):
        if message.role == "assistant":
            return message.content[0].text.value
    raise ValueError("No assistant message found.")

def parse_and_validate_notes(content):
    data = json.loads(content)
    notes = [Note(**item) for item in data["notes"]]
    return notes, data

def save_notes_to_file(data, filename="exam_notes.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n📝 Notes saved to {filename}")

def main():
    client = get_client()
    assistant_id = load_assistant_id()
    system_prompt = create_summary_prompt()

    print("📚 Creating thread and sending message...")
    thread_id = create_thread_and_message(client)

    print("⏳ Running assistant for structured summary...")
    run_assistant(client, assistant_id, thread_id, system_prompt)

    print("📥 Extracting response...")
    content = extract_response_text(client, thread_id)

    try:
        notes, raw_data = parse_and_validate_notes(content)
        print("✅ 10 valid notes generated:\n")
        for note in notes:
            print(f"{note.id}. {note.heading} — {note.summary}")
        save_notes_to_file(raw_data)
    except Exception as e:
        print("❌ Failed to parse or validate JSON:", e)
        print("\nRaw response:")
        print(content)


if __name__ == "__main__":
    main()
