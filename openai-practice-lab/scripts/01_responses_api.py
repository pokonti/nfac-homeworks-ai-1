#!/usr/bin/env python3
"""
01 — Responses API Lab

Walk-through of OpenAI Threads → Runs → streaming workflow.
Demonstrates message handling, run polling, streaming responses, and tool calls.

Usage: python scripts/01_responses_api.py

Docs: https://platform.openai.com/docs/api-reference/responses
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def get_client():
    """Initialize OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)
    
    org_id = os.getenv("OPENAI_ORG")
    client_kwargs = {"api_key": api_key}
    if org_id:
        client_kwargs["organization"] = org_id
    
    return OpenAI(**client_kwargs)

def load_assistant_id():
    """Load assistant ID from .assistant file."""
    assistant_file = Path(".assistant")
    if not assistant_file.exists():
        print("❌ No assistant found. Please run: python scripts/00_init_assistant.py")
        sys.exit(1)
    return assistant_file.read_text().strip()

def create_thread_with_messages(client):
    """Create a thread and add sample messages."""
    print("📝 Creating thread with messages...")
    
    # Create thread with initial messages
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Hello! I'm learning about the OpenAI Assistant API. Can you explain how threads and runs work?"
            },
            {
                "role": "user", 
                "content": "Also, what are the key benefits of using the Assistant API compared to the Chat Completions API?"
            }
        ]
    )
    
    print(f"✅ Thread created: {thread.id}")
    return thread

def demonstrate_polling_run(client, assistant_id, thread_id):
    """Demonstrate run creation with polling until completion."""
    print("\n🔄 Starting run with polling...")
    
    start_time = time.time()
    
    # Create and start run
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions="Please provide clear, educational explanations suitable for someone learning the API."
    )
    
    print(f"🚀 Run started: {run.id}")
    print(f"📊 Initial status: {run.status}")
    
    # Poll until completion
    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        print(f"⏳ Status: {run.status}")
        
        if run.status == "requires_action":
            print("🔧 Run requires action (tool calls)")
            # In a real scenario, you'd handle tool calls here
            break
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Run completed in {duration:.2f} seconds")
    print(f"📊 Final status: {run.status}")
    
    if run.usage:
        print(f"💰 Token usage: {run.usage.total_tokens} total "
              f"({run.usage.prompt_tokens} prompt + {run.usage.completion_tokens} completion)")
    
    return run

def demonstrate_streaming_run(client, assistant_id, thread_id):
    """Demonstrate streaming run with real-time token display."""
    print("\n🌊 Starting streaming run...")
    
    # Add another message to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content="Can you give me a practical example of when I might use file_search vs function calling?"
    )
    
    print("📡 Streaming response:")
    print("-" * 50)
    
    # Create streaming run
    stream = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        stream=True,
        instructions="Provide a concise but practical example with code snippets if helpful."
    )
    
    full_response = ""
    for event in stream:
        if event.event == "thread.message.delta":
            if hasattr(event.data.delta, 'content') and event.data.delta.content:
                for content in event.data.delta.content:
                    if hasattr(content, 'text') and hasattr(content.text, 'value'):
                        chunk = content.text.value
                        print(chunk, end="", flush=True)
                        full_response += chunk
        elif event.event == "thread.run.completed":
            print(f"\n\n✅ Streaming completed")
            if hasattr(event.data, 'usage') and event.data.usage:
                usage = event.data.usage
                print(f"💰 Token usage: {usage.total_tokens} total "
                      f"({usage.prompt_tokens} prompt + {usage.completion_tokens} completion)")
    
    print("-" * 50)
    return full_response

def retrieve_thread_messages(client, thread_id):
    """Retrieve and display all messages in the thread."""
    print("\n📋 Thread conversation history:")
    print("=" * 50)
    
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    
    for message in reversed(messages.data):
        role = message.role.upper()
        content = ""
        
        for content_block in message.content:
            if content_block.type == "text":
                content = content_block.text.value
                break
        
        print(f"\n{role}:")
        print(content[:500] + ("..." if len(content) > 500 else ""))
    
    print("=" * 50)

def demonstrate_run_steps(client, thread_id, run_id):
    """Retrieve and display run steps for debugging."""
    print(f"\n🔍 Analyzing run steps for run: {run_id}")
    
    try:
        steps = client.beta.threads.runs.steps.list(thread_id=thread_id, run_id=run_id)
        
        print(f"📊 Found {len(steps.data)} steps:")
        
        for i, step in enumerate(steps.data, 1):
            print(f"\n  Step {i}: {step.type}")
            print(f"    Status: {step.status}")
            print(f"    Created: {step.created_at}")
            
            if step.type == "tool_calls" and step.step_details:
                print(f"    Tool calls: {len(step.step_details.tool_calls)}")
                for tool_call in step.step_details.tool_calls:
                    print(f"      - {tool_call.type}")
    
    except Exception as e:
        print(f"⚠️  Could not retrieve run steps: {e}")

def main():
    """Main function to run the Responses API lab."""
    print("🚀 OpenAI Practice Lab - Responses API")
    print("=" * 50)
    
    # Initialize client and get assistant
    client = get_client()
    assistant_id = load_assistant_id()
    print(f"✅ Using assistant: {assistant_id}")
    
    # 1. Create thread with messages
    thread = create_thread_with_messages(client)
    
    # 2. Demonstrate polling run
    run = demonstrate_polling_run(client, assistant_id, thread.id)
    
    # 3. Show run steps for debugging
    demonstrate_run_steps(client, thread.id, run.id)
    
    # 4. Demonstrate streaming run
    demonstrate_streaming_run(client, assistant_id, thread.id)
    
    # 5. Show final conversation
    retrieve_thread_messages(client, thread.id)
    
    # Save thread ID for potential cleanup
    thread_file = Path(".last_thread")
    thread_file.write_text(thread.id)
    
    print(f"\n🎯 Lab Complete!")
    print(f"   Thread ID saved to: {thread_file}")
    print(f"   Next: python scripts/02_structured_output.py")
    print(f"   Cleanup: python scripts/99_cleanup.py")

if __name__ == "__main__":
    main() 