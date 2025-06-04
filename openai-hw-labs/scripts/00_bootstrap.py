import os
import sys
import requests
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load env variables
load_dotenv()

# Init client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY is missing in your .env file")
    sys.exit(1)

client = OpenAI(api_key=api_key)

def load_assistant_id():
    """Load existing assistant ID from .assistant file if it exists."""
    assistant_file = Path(".assistant")
    if assistant_file.exists():
        return assistant_file.read_text().strip()
    return None

def save_assistant_id(assistant_id):
    with open(".assistant", "w") as f:
        f.write(assistant_id)
    print(f"💾 Assistant ID saved to .assistant")


def create_or_update_assistant(client):
    """Create a new assistant or update existing one."""
    existing_id = load_assistant_id()

    assistant_config = {
        "name": "Study Q&A Assistant",
        "instructions": "You are a helpful tutor. Use the attached files to answer questions. Cite sources.",
        "model": "gpt-4o-mini",
        "tools": [{"type": "file_search"}]
    }

    try:
        if existing_id:
            print(f"🔄 Updating existing assistant: {existing_id}")
            assistant = client.beta.assistants.update(
                assistant_id=existing_id,
                **assistant_config
            )
            print("✅ Assistant updated successfully!")
        else:
            print("🆕 Creating new assistant...")
            assistant = client.beta.assistants.create(**assistant_config)
            save_assistant_id(assistant.id)
            print("✅ Assistant created successfully!")

        print(f"📋 Assistant Details:")
        print(f"   ID: {assistant.id}")
        print(f"   Name: {assistant.name}")
        print(f"   Model: {assistant.model}")
        print(f"   Tools: {[tool.type for tool in assistant.tools]}")

        return assistant

    except Exception as e:
        print(f"❌ Error creating/updating assistant: {e}")
        sys.exit(1)

def create_file(client, file_path):
    """Upload file from local path or URL"""
    if file_path.startswith("http://") or file_path.startswith("https://"):
        response = requests.get(file_path)
        file_content = BytesIO(response.content)
        file_name = file_path.split("/")[-1]
        result = client.files.create(
            file=(file_name, file_content),
            purpose="assistants"
        )
    else:
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="assistants"
            )
    print(f"📎 Uploaded file: {result.id}")
    return result.id


def main():
    # 1. Create Assistant
    print("🧠 Creating assistant...")
    assistant = create_or_update_assistant(client)
    # print(f"✅ Assistant created: {assistant.id}")

    # 2. Upload file (change this path or URL)
    file_path = "../data/Cognitive_science.pdf"
    file_id = create_file(client, file_path)

    # 3. Create vector store and add the file
    print("📚 Creating vector store...")
    vector_store = client.vector_stores.create(
        name="knowledge_base"
    )
    print(vector_store.id)

    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_id
    )
    result = client.vector_stores.files.list(
        vector_store_id=vector_store.id
    )
    print(result)

    print(f"✅ File added to vector store: {vector_store.id}")

    # 4. Link vector store to assistant
    print("🔗 Linking vector store to assistant...")
    client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id]
            }
        }
    )

    print("✅ Assistant ready with file knowledge")

    # 5. Save assistant ID
    save_assistant_id(assistant.id)


if __name__ == "__main__":
    main()
