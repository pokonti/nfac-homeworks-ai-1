import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)

    org_id = os.getenv("OPENAI_ORG")
    client_kwargs = {"api_key": api_key}
    if org_id:
        client_kwargs["organization"] = org_id

    return OpenAI(**client_kwargs)


def load_assistant_id():
    assistant_file = Path(".assistant")
    if not assistant_file.exists():
        print("❌ No assistant found. Please run: python scripts/00_bootstrap.py")
        sys.exit(1)
    return assistant_file.read_text().strip()


def ask_pdf_question(client, assistant_id, question):
    print(f"\n📝 Asking: {question}")

    # Create a new thread
    thread = client.beta.threads.create()

    # Add the user question
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question
    )

    # Create and start the run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        instructions="Answer using attached files. Cite sources if possible."
    )

    # Poll until completed
    print("⏳ Waiting for response...")
    while run.status not in ["completed", "failed", "cancelled"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        print(f"📡 Status: {run.status}")

    if run.status != "completed":
        print(f"❌ Run did not complete successfully: {run.status}")
        return

    # Get and print response with citations
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    for message in reversed(messages.data):
        if message.role == "assistant":
            print("\n📘 Answer:")
            print(message.content[0].text.value)

            annotations = message.content[0].text.annotations
            if annotations:
                print("\n🔍 Citations (from PDF):")
                for ann in annotations:
                    print(f"- {ann}")
            else:
                print("\n⚠️ No citations found — did not reference uploaded PDF.")
            break


def main():
    print("📚 Assistant Q&A from PDF")
    print("=" * 40)

    client = get_client()
    assistant_id = load_assistant_id()
    print(f"✅ Using assistant: {assistant_id}")

    # Example prompts from your homework
    ask_pdf_question(client, assistant_id,
                     "How do cortisol levels correlate with anxiety and depression according to the UK Biobank study?")

    ask_pdf_question(client, assistant_id,
                     "What machine learning models are proposed for analyzing voice, facial expression, and physiological data in the study?")

    print("\n🎯 Done! You can now verify if responses referenced chunk IDs.")


if __name__ == "__main__":
    main()
