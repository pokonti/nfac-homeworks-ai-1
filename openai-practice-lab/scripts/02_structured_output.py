#!/usr/bin/env python3
"""
02 — Structured Output Lab

Demonstrates guaranteed JSON output matching Pydantic models.
Compares JSON-mode vs function tools with "strict": True schema validation.

Usage: python scripts/02_structured_output.py

Docs: https://platform.openai.com/docs/guides/structured-output
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Pydantic Models for Structured Output
class WeatherAlert(BaseModel):
    """Weather alert information with structured fields."""
    location: str = Field(description="Geographic location of the alert")
    severity: str = Field(description="Alert severity: low, medium, high, critical")
    alert_type: str = Field(description="Type of weather alert")
    description: str = Field(description="Detailed description of the weather condition")
    advice: str = Field(description="Recommended actions for safety")
    expires_at: Optional[str] = Field(description="When the alert expires (if known)")

class TechAnalysis(BaseModel):
    """Technical analysis of a programming concept."""
    concept: str = Field(description="The programming concept being analyzed")
    difficulty_level: str = Field(description="Beginner, Intermediate, or Advanced")
    key_benefits: List[str] = Field(description="Main advantages of this concept")
    common_pitfalls: List[str] = Field(description="Common mistakes to avoid")
    use_cases: List[str] = Field(description="Practical applications")
    learning_resources: List[str] = Field(description="Recommended learning materials")

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

def demonstrate_json_mode(client, assistant_id):
    """Demonstrate basic JSON mode without strict schema validation."""
    print("🔧 Demonstrating JSON Mode (Basic)")
    print("-" * 40)
    
    # Create thread for JSON mode demo
    thread = client.beta.threads.create(
        messages=[{
            "role": "user",
            "content": """Create a weather alert for a severe thunderstorm in Chicago. 
            Return the response as a JSON object with fields: location, severity, alert_type, 
            description, advice, and expires_at."""
        }]
    )
    
    # Run with JSON mode
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        response_format={"type": "json_object"},
        instructions="Always respond with valid JSON. Use clear, structured data."
    )
    
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        response_content = messages.data[0].content[0].text.value
        
        print("📄 Raw JSON Response:")
        print(response_content)
        
        try:
            json_data = json.loads(response_content)
            print("\n✅ Valid JSON parsed successfully")
            print(f"📊 Fields: {list(json_data.keys())}")
            
            # Try to validate with Pydantic (may fail due to loose schema)
            try:
                weather_alert = WeatherAlert(**json_data)
                print("✅ Pydantic validation successful!")
                return weather_alert
            except Exception as e:
                print(f"⚠️  Pydantic validation failed: {e}")
                return json_data
                
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return None
    else:
        print(f"❌ Run failed with status: {run.status}")
        return None

def demonstrate_function_tools_strict(client, assistant_id):
    """Demonstrate function tools with strict schema validation."""
    print("\n🎯 Demonstrating Function Tools (Strict Schema)")
    print("-" * 50)
    
    # Create assistant with function tool
    function_schema = {
        "name": "analyze_tech_concept",
        "description": "Analyze a programming or technology concept",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "The programming concept being analyzed"
                },
                "difficulty_level": {
                    "type": "string",
                    "enum": ["Beginner", "Intermediate", "Advanced"],
                    "description": "Difficulty level"
                },
                "key_benefits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Main advantages of this concept"
                },
                "common_pitfalls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Common mistakes to avoid"
                },
                "use_cases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Practical applications"
                },
                "learning_resources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recommended learning materials"
                }
            },
            "required": ["concept", "difficulty_level", "key_benefits", "common_pitfalls", "use_cases"],
            "additionalProperties": False
        }
    }
    
    # Update assistant with function tool
    client.beta.assistants.update(
        assistant_id=assistant_id,
        tools=[
            {"type": "file_search"},
            {"type": "function", "function": function_schema}
        ]
    )
    
    # Create thread for function demo
    thread = client.beta.threads.create(
        messages=[{
            "role": "user",
            "content": "Please analyze the concept of 'Async/Await in Python' using the analyze_tech_concept function."
        }]
    )
    
    # Run with function calling
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        instructions="Use the analyze_tech_concept function to provide a structured analysis."
    )
    
    if run.status == "completed":
        # Get run steps to find function calls
        steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
        
        for step in steps.data:
            if step.type == "tool_calls":
                for tool_call in step.step_details.tool_calls:
                    if tool_call.type == "function":
                        function_args = json.loads(tool_call.function.arguments)
                        
                        print("📋 Function Call Arguments:")
                        print(json.dumps(function_args, indent=2))
                        
                        # Validate with Pydantic
                        try:
                            tech_analysis = TechAnalysis(**function_args)
                            print("\n✅ Strict schema validation successful!")
                            print(f"📊 Concept: {tech_analysis.concept}")
                            print(f"📊 Difficulty: {tech_analysis.difficulty_level}")
                            print(f"📊 Benefits: {len(tech_analysis.key_benefits)} items")
                            print(f"📊 Pitfalls: {len(tech_analysis.common_pitfalls)} items")
                            return tech_analysis
                        except Exception as e:
                            print(f"❌ Pydantic validation failed: {e}")
                            return function_args
        
        print("⚠️  No function calls found in run steps")
        return None
    else:
        print(f"❌ Run failed with status: {run.status}")
        return None

def compare_approaches(json_result, function_result):
    """Compare the results from both approaches."""
    print("\n📊 Comparison of Approaches")
    print("=" * 50)
    
    print("🔧 JSON Mode:")
    if json_result:
        if isinstance(json_result, WeatherAlert):
            print("  ✅ Pydantic validation: SUCCESS")
            print(f"  📍 Location: {json_result.location}")
            print(f"  ⚠️  Severity: {json_result.severity}")
        else:
            print("  ⚠️  Pydantic validation: FAILED (loose schema)")
            print(f"  📄 Raw data type: {type(json_result)}")
    else:
        print("  ❌ No valid result")
    
    print("\n🎯 Function Tools (Strict):")
    if function_result:
        if isinstance(function_result, TechAnalysis):
            print("  ✅ Pydantic validation: SUCCESS")
            print(f"  🎓 Concept: {function_result.concept}")
            print(f"  📈 Difficulty: {function_result.difficulty_level}")
        else:
            print("  ⚠️  Pydantic validation: FAILED")
            print(f"  📄 Raw data type: {type(function_result)}")
    else:
        print("  ❌ No valid result")
    
    print("\n💡 Key Takeaways:")
    print("  • JSON Mode: Flexible but may not match exact schema")
    print("  • Function Tools (Strict): Guaranteed schema compliance")
    print("  • Use Function Tools for production applications requiring exact structure")

def reset_assistant_tools(client, assistant_id):
    """Reset assistant tools to original state."""
    client.beta.assistants.update(
        assistant_id=assistant_id,
        tools=[{"type": "file_search"}]
    )

def main():
    """Main function to run the structured output lab."""
    print("🚀 OpenAI Practice Lab - Structured Output")
    print("=" * 50)
    
    # Initialize client and get assistant
    client = get_client()
    assistant_id = load_assistant_id()
    print(f"✅ Using assistant: {assistant_id}")
    
    try:
        # 1. Demonstrate JSON mode
        json_result = demonstrate_json_mode(client, assistant_id)
        
        # 2. Demonstrate function tools with strict schema
        function_result = demonstrate_function_tools_strict(client, assistant_id)
        
        # 3. Compare approaches
        compare_approaches(json_result, function_result)
        
        print(f"\n🎯 Lab Complete!")
        print(f"   Next: python scripts/03_rag_file_search.py")
        print(f"   Cleanup: python scripts/99_cleanup.py")
        
    finally:
        # Reset assistant tools to original state
        reset_assistant_tools(client, assistant_id)
        print("🔄 Assistant tools reset to original state")

if __name__ == "__main__":
    main() 