#!/usr/bin/env python3
"""
03 — RAG via file_search Lab

End-to-end RAG demonstration using OpenAI's built-in file_search tool.
No external vector DB required - OpenAI hosts the vector store.

Usage: python scripts/03_rag_file_search.py

Docs: https://platform.openai.com/docs/tools/file-search
"""

import os
import sys
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

def create_sample_documents():
    """Create sample documents for RAG demonstration."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample document 1: Introduction to LLMs
    llm_intro = """# Introduction to Large Language Models (LLMs)

## What are Large Language Models?

Large Language Models (LLMs) are artificial intelligence systems trained on vast amounts of text data to understand and generate human-like text. They use transformer architecture and attention mechanisms to process and generate language.

## Key Characteristics

### Scale
- Billions to trillions of parameters
- Trained on diverse internet text
- Require significant computational resources

### Capabilities
- Text generation and completion
- Question answering
- Language translation
- Code generation
- Summarization
- Creative writing

## Popular LLM Models

### GPT Series (OpenAI)
- GPT-3: 175 billion parameters
- GPT-4: Multimodal capabilities
- GPT-4o: Optimized for speed and efficiency

### Other Notable Models
- Claude (Anthropic): Focus on safety and helpfulness
- LLaMA (Meta): Open-source approach
- PaLM (Google): Pathways Language Model

## Applications

### Business Use Cases
- Customer service chatbots
- Content creation and marketing
- Code assistance and debugging
- Document analysis and summarization

### Educational Applications
- Tutoring and explanation
- Language learning assistance
- Research and writing support

## Limitations and Considerations

### Technical Limitations
- Hallucination: generating false information
- Context window limitations
- Training data cutoff dates
- Computational requirements

### Ethical Considerations
- Bias in training data
- Privacy and data security
- Job displacement concerns
- Misinformation potential

## Best Practices

### Prompt Engineering
- Be specific and clear in requests
- Provide context and examples
- Use iterative refinement
- Test different approaches

### Safety and Reliability
- Verify important information
- Use multiple sources
- Implement human oversight
- Regular model evaluation

## Future Directions

The field of LLMs continues to evolve with:
- Improved efficiency and smaller models
- Better reasoning capabilities
- Multimodal integration (text, image, audio)
- Specialized domain models
- Enhanced safety measures
"""

    # Sample document 2: API Best Practices
    api_practices = """# API Integration Best Practices

## Authentication and Security

### API Key Management
- Store keys in environment variables
- Use different keys for development/production
- Rotate keys regularly
- Never commit keys to version control

### Rate Limiting
- Implement exponential backoff
- Monitor usage quotas
- Cache responses when appropriate
- Use batch operations when available

## Error Handling

### Common HTTP Status Codes
- 200: Success
- 400: Bad Request (client error)
- 401: Unauthorized (authentication issue)
- 429: Too Many Requests (rate limit)
- 500: Internal Server Error

### Retry Strategies
- Implement retry logic for transient errors
- Use exponential backoff with jitter
- Set maximum retry limits
- Log errors for debugging

## Performance Optimization

### Request Optimization
- Minimize payload size
- Use appropriate HTTP methods
- Implement request compression
- Batch multiple operations

### Response Handling
- Stream large responses
- Parse JSON efficiently
- Cache frequently accessed data
- Implement pagination for large datasets

## Monitoring and Logging

### Key Metrics
- Response times
- Error rates
- Usage patterns
- Cost tracking

### Logging Best Practices
- Log request/response metadata
- Include correlation IDs
- Sanitize sensitive data
- Use structured logging formats

## OpenAI API Specific Tips

### Model Selection
- Use gpt-4o-mini for cost-effective tasks
- Use gpt-4 for complex reasoning
- Consider fine-tuned models for specialized tasks

### Token Management
- Monitor token usage
- Optimize prompt length
- Use system messages effectively
- Implement token counting

### Assistant API
- Reuse assistants across sessions
- Manage thread lifecycle
- Use file_search for RAG applications
- Clean up resources regularly

## Testing Strategies

### Unit Testing
- Mock API responses
- Test error conditions
- Validate input/output formats
- Test rate limiting behavior

### Integration Testing
- Test with real API endpoints
- Validate end-to-end workflows
- Test with various data sizes
- Monitor performance metrics

## Documentation

### API Documentation
- Keep documentation up-to-date
- Include code examples
- Document error scenarios
- Provide troubleshooting guides

### Code Documentation
- Comment complex logic
- Document configuration options
- Include usage examples
- Maintain changelog
"""

    # Write sample documents
    (data_dir / "intro_to_llms.md").write_text(llm_intro)
    (data_dir / "api_best_practices.md").write_text(api_practices)
    
    print(f"📄 Created sample documents in {data_dir}/")
    return [
        data_dir / "intro_to_llms.md",
        data_dir / "api_best_practices.md"
    ]

def upload_documents(client, file_paths):
    """Upload documents for knowledge retrieval."""
    print("📤 Uploading documents...")
    
    uploaded_files = []
    for file_path in file_paths:
        print(f"  Uploading: {file_path.name}")
        
        with open(file_path, "rb") as file:
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
        
        uploaded_files.append(uploaded_file)
        print(f"  ✅ File ID: {uploaded_file.id}")
    
    return uploaded_files

def create_vector_store(client, uploaded_files):
    """Create a vector store and add files to it."""
    print("\n🗂️  Creating vector store...")
    
    # Create vector store
    vector_store = client.beta.vector_stores.create(
        name="Practice Lab Knowledge Base",
        expires_after={
            "anchor": "last_active_at",
            "days": 7
        }
    )
    
    print(f"✅ Vector store created: {vector_store.id}")
    
    # Add files to vector store
    file_batch = client.beta.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store.id,
        file_ids=[file.id for file in uploaded_files]
    )
    
    print(f"📊 File batch status: {file_batch.status}")
    print(f"📊 Files processed: {file_batch.file_counts.completed}/{file_batch.file_counts.total}")
    
    return vector_store

def attach_vector_store_to_assistant(client, assistant_id, vector_store_id):
    """Attach vector store to assistant for file_search."""
    print(f"\n🔗 Attaching vector store to assistant...")
    
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        }
    )
    
    print("✅ Vector store attached to assistant")
    return assistant

def demonstrate_rag_queries(client, assistant_id):
    """Demonstrate RAG queries with file_search."""
    print("\n🔍 Demonstrating RAG Queries")
    print("=" * 40)
    
    queries = [
        "What are the key characteristics of Large Language Models?",
        "What are the best practices for API key management?",
        "How should I handle rate limiting when using APIs?",
        "What are the limitations of LLMs that I should be aware of?",
        "Can you compare different LLM models mentioned in the documents?"
    ]
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 50)
        
        # Create thread for this query
        thread = client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": f"{query}\n\nPlease provide a comprehensive answer based on the uploaded documents and include specific citations."
            }]
        )
        
        # Run with file_search
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="Use the file_search tool to find relevant information from the uploaded documents. Always cite your sources and provide specific references."
        )
        
        if run.status == "completed":
            # Get the response
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            response = messages.data[0].content[0].text.value
            
            print("🤖 Assistant Response:")
            print(response[:300] + ("..." if len(response) > 300 else ""))
            
            # Check for citations
            if hasattr(messages.data[0].content[0].text, 'annotations'):
                annotations = messages.data[0].content[0].text.annotations
                if annotations:
                    print(f"\n📚 Citations found: {len(annotations)}")
                    for j, annotation in enumerate(annotations[:3], 1):  # Show first 3
                        if hasattr(annotation, 'file_citation'):
                            print(f"  {j}. File: {annotation.file_citation.file_id}")
            
            # Analyze run steps for file_search usage
            steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
            file_search_used = False
            
            for step in steps.data:
                if step.type == "tool_calls":
                    for tool_call in step.step_details.tool_calls:
                        if tool_call.type == "file_search":
                            file_search_used = True
                            print("🔍 file_search tool was used")
                            break
            
            if not file_search_used:
                print("⚠️  file_search tool was not used")
            
            results.append({
                "query": query,
                "response_length": len(response),
                "file_search_used": file_search_used,
                "thread_id": thread.id
            })
        
        else:
            print(f"❌ Query failed with status: {run.status}")
            results.append({
                "query": query,
                "status": run.status,
                "thread_id": thread.id
            })
    
    return results

def analyze_rag_performance(results):
    """Analyze the performance of RAG queries."""
    print("\n📊 RAG Performance Analysis")
    print("=" * 50)
    
    successful_queries = [r for r in results if "response_length" in r]
    
    print(f"✅ Successful queries: {len(successful_queries)}/{len(results)}")
    
    if successful_queries:
        avg_response_length = sum(r["response_length"] for r in successful_queries) / len(successful_queries)
        file_search_usage = sum(1 for r in successful_queries if r["file_search_used"])
        
        print(f"📏 Average response length: {avg_response_length:.0f} characters")
        print(f"🔍 file_search usage: {file_search_usage}/{len(successful_queries)} queries")
        
        print("\n💡 Key Insights:")
        print("  • file_search automatically retrieves relevant document chunks")
        print("  • Citations provide traceability to source documents")
        print("  • Response quality depends on document content and query specificity")
        print("  • Vector search handles semantic similarity well")

def cleanup_resources(client, uploaded_files, vector_store_id):
    """Clean up uploaded files and vector store."""
    print("\n🧹 Cleaning up resources...")
    
    # Delete uploaded files
    for file in uploaded_files:
        try:
            client.files.delete(file.id)
            print(f"🗑️  Deleted file: {file.id}")
        except Exception as e:
            print(f"⚠️  Could not delete file {file.id}: {e}")
    
    # Delete vector store
    try:
        client.beta.vector_stores.delete(vector_store_id)
        print(f"🗑️  Deleted vector store: {vector_store_id}")
    except Exception as e:
        print(f"⚠️  Could not delete vector store {vector_store_id}: {e}")

def main():
    """Main function to run the RAG file_search lab."""
    print("🚀 OpenAI Practice Lab - RAG with file_search")
    print("=" * 50)
    
    # Initialize client and get assistant
    client = get_client()
    assistant_id = load_assistant_id()
    print(f"✅ Using assistant: {assistant_id}")
    
    uploaded_files = None
    vector_store = None
    
    try:
        # 1. Create sample documents
        file_paths = create_sample_documents()
        
        # 2. Upload documents
        uploaded_files = upload_documents(client, file_paths)
        
        # 3. Create vector store
        vector_store = create_vector_store(client, uploaded_files)
        
        # 4. Attach vector store to assistant
        attach_vector_store_to_assistant(client, assistant_id, vector_store.id)
        
        # 5. Demonstrate RAG queries
        results = demonstrate_rag_queries(client, assistant_id)
        
        # 6. Analyze performance
        analyze_rag_performance(results)
        
        print(f"\n🎯 Lab Complete!")
        print(f"   Vector store will auto-expire in 7 days")
        print(f"   Next: Explore run logs or extend with more documents")
        print(f"   Cleanup: python scripts/99_cleanup.py")
        
    except Exception as e:
        print(f"❌ Error during RAG lab: {e}")
        
    finally:
        # Optional: Clean up resources immediately
        cleanup_choice = input("\n🤔 Clean up resources now? (y/N): ").lower().strip()
        if cleanup_choice == 'y' and uploaded_files and vector_store:
            cleanup_resources(client, uploaded_files, vector_store.id)

if __name__ == "__main__":
    main() 