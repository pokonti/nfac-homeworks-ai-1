import os
import time
import pytest
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope="module")
def client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.fail("OPENAI_API_KEY is not set in the .env file")
    return OpenAI(api_key=api_key)

@pytest.fixture(scope="module")
def assistant_id():
    path = Path(".assistant")
    if not path.exists():
        pytest.fail(".assistant file is missing. Run 00_bootstrap.py first.")
    return path.read_text().strip()

def ask_and_get_annotations(client, assistant_id, question):
    # Create a thread
    thread = client.beta.threads.create()

    # Add the question
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question
    )

    # Run assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        instructions="Answer using the uploaded file. Cite the source if applicable."
    )

    # Poll for result
    while run.status not in ["completed", "failed", "cancelled"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    assert run.status == "completed", f"Run did not complete successfully: {run.status}"

    # Get response
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    for message in reversed(messages.data):
        if message.role == "assistant":
            text = message.content[0].text
            return text.value, text.annotations

    return "", []

@pytest.mark.parametrize("question", [
    "How do cortisol levels correlate with anxiety and depression according to the UK Biobank study?",
    "What machine learning models are proposed for analyzing voice, facial expression, and physiological data in the study?"
])
def test_pdf_citations_present(client, assistant_id, question):
    response, annotations = ask_and_get_annotations(client, assistant_id, question)

    assert response.strip() != "", "Assistant returned an empty response"
    assert annotations, "No citations found — assistant did not use the uploaded PDF"
    assert any("chunk_id" in str(a) for a in annotations), "No chunk_id found in annotations"
