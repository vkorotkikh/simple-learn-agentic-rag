"""
Simple Example Usage of the Agentic RAG System
This script demonstrates how to quickly set up and use the agentic RAG system.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_rag import AgenticRAG, ChunkingStrategy
from config_utils import RAGConfig, DocumentLoader, PerformanceMonitor, ResponseCache
import time

def simple_example():
    """Simple example with built-in sample documents"""
    print("="*80)
    print("🚀 SIMPLE AGENTIC RAG EXAMPLE")
    print("="*80)
    
    # Initialize the system
    print("\n📦 Initializing Agentic RAG System...")
    rag = AgenticRAG(
        llm_model="llama2",  # Change this to your preferred Ollama model
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunking_strategy=ChunkingStrategy.HYBRID
    )
    
    # Add sample knowledge base
    print("\n📚 Loading knowledge base...")
    knowledge_base = [
        """
        Retrieval-Augmented Generation (RAG) is an AI framework that combines the power of 
        pre-trained language models with retrieval mechanisms. Instead of relying solely on 
        the parametric knowledge encoded in the model weights, RAG systems retrieve relevant 
        information from external knowledge sources and use this context to generate more 
        accurate and factual responses. This approach significantly reduces hallucinations 
        and allows the system to access up-to-date information.
        """,
        
        """
        Agentic RAG systems use multiple specialized agents to handle different aspects of 
        the retrieval and generation process. The Query Router agent analyzes incoming queries 
        to understand intent and complexity. The Retriever agent searches through the knowledge 
        base using advanced chunking strategies. The Reranker agent uses cross-encoder models 
        to reorder retrieved documents by relevance. Finally, the Synthesizer agent combines 
        the retrieved context to generate comprehensive answers.
        """,
        
        """
        Semantic chunking is an advanced text splitting technique that divides documents based 
        on meaning rather than fixed character counts. It uses embedding models to identify 
        natural breakpoints in the text where the semantic meaning shifts. This results in 
        more coherent chunks that preserve context better than traditional recursive splitting. 
        Hybrid approaches combine both semantic and recursive chunking to leverage the benefits 
        of each method.
        """,
        
        """
        RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG 
        pipelines. It provides metrics like faithfulness (whether the answer is grounded in 
        the context), answer relevancy (how relevant the answer is to the question), context 
        precision (the relevance of retrieved contexts), and context recall (whether all 
        relevant information was retrieved). These metrics help developers understand and 
        improve their RAG systems.
        """,
        
        """
        Cross-encoder reranking is a technique used to improve the relevance of retrieved 
        documents. Unlike bi-encoders that encode queries and documents separately, 
        cross-encoders process the query and document together, allowing for more nuanced 
        understanding of relevance. This comes at a computational cost, so cross-encoders 
        are typically used to rerank a smaller set of candidates retrieved by faster methods.
        """
    ]
    
    rag.add_documents(knowledge_base)
    print(f"✅ Loaded {len(knowledge_base)} documents into the knowledge base")
    
    # Example queries
    queries = [
        "What is RAG and how does it work?",
        "Explain the role of different agents in an agentic RAG system",
        "What is semantic chunking and why is it useful?",
        "How can we evaluate RAG system performance?",
        "What is the difference between cross-encoders and bi-encoders in reranking?"
    ]
    
    print("\n" + "="*80)
    print("🔍 PROCESSING QUERIES")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*60}")
        print(f"Query {i}: {query}")
        print(f"{'─'*60}")
        
        # Process query
        start_time = time.time()
        response = rag.process_query(query)
        elapsed_time = time.time() - start_time
        
        # Display results
        print(f"\n💡 Answer:")
        print(response.answer)
        
        print(f"\n📊 Metrics:")
        print(f"  • Confidence: {response.confidence:.2%}")
        print(f"  • Response Time: {elapsed_time:.2f}s")
        print(f"  • Contexts Used: {len(response.contexts)}")
        print(f"  • Agent Steps: {len(response.agent_trace)}")
        
        if response.evaluation_metrics:
            print(f"\n📈 Evaluation Scores:")
            for metric, score in response.evaluation_metrics.items():
                if isinstance(score, (int, float)):
                    print(f"  • {metric}: {score:.3f}")

def advanced_example():
    """Advanced example with custom configuration and document loading"""
    print("\n" + "="*80)
    print("🔧 ADVANCED AGENTIC RAG EXAMPLE")
    print("="*80)
    
    # Create custom configuration
    print("\n⚙️ Creating custom configuration...")
    config = RAGConfig(
        llm={"model_name": "llama2", "temperature": 0.2},
        embedding={"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
        chunking={
            "strategy": "hybrid",
            "chunk_size": 600,
            "chunk_overlap": 150
        },
        reranker={"top_k": 3},
        num_agents=4,
        enable_evaluation=True,
        cache_enabled=True
    )
    
    # Save configuration
    config_path = "my_config.yaml"
    config.save(config_path)
    print(f"✅ Configuration saved to {config_path}")
    
    # Initialize with configuration
    print("\n📦 Initializing system with custom configuration...")
    rag = AgenticRAG(
        llm_model=config.llm.model_name,
        embedding_model=config.embedding.model_name,
        chunking_strategy=ChunkingStrategy(config.chunking.strategy)
    )
    
    # Initialize performance monitoring
    monitor = PerformanceMonitor()
    
    # Initialize caching
    cache = ResponseCache(cache_ttl=3600) if config.cache_enabled else None
    
    # Load documents from file (create a sample file first)
    sample_file = Path("sample_document.txt")
    sample_content = """
    Advanced Natural Language Processing Techniques
    
    Transformer architectures have revolutionized NLP by introducing self-attention mechanisms 
    that allow models to process sequences in parallel rather than sequentially. This breakthrough 
    led to the development of models like BERT, GPT, and T5.
    
    Fine-tuning strategies allow pre-trained models to be adapted for specific tasks with 
    relatively small amounts of task-specific data. Techniques like LoRA (Low-Rank Adaptation) 
    and QLoRA make fine-tuning more efficient by updating only a small subset of parameters.
    
    Prompt engineering has emerged as a crucial skill for working with large language models. 
    Effective prompts can dramatically improve model performance without any fine-tuning. 
    Techniques include few-shot learning, chain-of-thought prompting, and retrieval-augmented 
    generation.
    """
    
    sample_file.write_text(sample_content)
    print(f"\n📝 Created sample document: {sample_file}")
    
    # Load document
    loader = DocumentLoader()
    documents = loader.load_document(sample_file)
    rag.add_documents([doc.page_content for doc in documents])
    print(f"✅ Loaded document with {len(documents)} chunks")
    
    # Process queries with caching
    test_query = "What are transformer architectures and why are they important?"
    
    print(f"\n🔍 Query: {test_query}")
    
    # First query (cache miss)
    print("\n⏱️ First execution (no cache)...")
    start_time = time.time()
    
    cached_response = cache.get(test_query) if cache else None
    if cached_response:
        print("  Cache hit!")
        response = cached_response
    else:
        response = rag.process_query(test_query)
        if cache:
            cache.set(test_query, response.dict())
            monitor.record_cache_miss()
    
    elapsed_time = time.time() - start_time
    monitor.record_query(elapsed_time, response.confidence)
    
    print(f"  Response time: {elapsed_time:.2f}s")
    print(f"\n💡 Answer: {response.answer[:200]}...")
    
    # Second query (cache hit)
    if cache:
        print("\n⏱️ Second execution (with cache)...")
        start_time = time.time()
        
        cached_response = cache.get(test_query)
        if cached_response:
            print("  Cache hit! ✅")
            monitor.record_cache_hit()
        
        elapsed_time = time.time() - start_time
        print(f"  Response time: {elapsed_time:.4f}s (much faster!)")
    
    # Display performance summary
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    print(monitor.get_summary())
    
    # Cleanup
    sample_file.unlink()
    if Path(config_path).exists():
        Path(config_path).unlink()
    print("\n🧹 Cleanup completed")

def interactive_mode():
    """Interactive mode for custom queries"""
    print("\n" + "="*80)
    print("💬 INTERACTIVE MODE")
    print("="*80)
    
    print("\n📦 Initializing system...")
    rag = AgenticRAG(
        llm_model="llama2",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunking_strategy=ChunkingStrategy.HYBRID
    )
    
    print("\n📝 You can now:")
    print("  1. Add documents by typing: ADD <your text here>")
    print("  2. Ask questions by typing your query")
    print("  3. Exit by typing: quit, exit, or q")
    
    print("\nAdding default knowledge base...")
    default_docs = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing allows computers to understand and generate human language."
    ]
    rag.add_documents(default_docs)
    print(f"✅ Added {len(default_docs)} default documents")
    
    while True:
        user_input = input("\n🤔 Your input: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.upper().startswith("ADD "):
            # Add document
            doc_text = user_input[4:].strip()
            if doc_text:
                rag.add_documents([doc_text])
                print("✅ Document added to knowledge base")
            else:
                print("❌ Please provide text after ADD")
        
        elif user_input:
            # Process query
            print("\n⏳ Processing...")
            try:
                response = rag.process_query(user_input)
                print(f"\n💡 Answer:")
                print(response.answer)
                print(f"\n📊 Confidence: {response.confidence:.2%}")
                print(f"📚 Used {len(response.contexts)} contexts")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Please enter a query or command")

def main():
    """Main entry point with menu"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        🤖 AGENTIC RAG SYSTEM - EXAMPLES 🤖          ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 Available Examples:")
    print("  1. Simple Example - Quick demonstration with built-in documents")
    print("  2. Advanced Example - Custom config, file loading, and caching")
    print("  3. Interactive Mode - Add documents and ask questions interactively")
    print("  4. Exit")
    
    while True:
        choice = input("\n👉 Select an option (1-4): ").strip()
        
        if choice == '1':
            simple_example()
        elif choice == '2':
            advanced_example()
        elif choice == '3':
            interactive_mode()
        elif choice == '4':
            print("\n👋 Thank you for using the Agentic RAG System!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")
    
    print("\n✨ Examples completed successfully!")

if __name__ == "__main__":
    # Check if Ollama is available
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Ollama not running")
    except:
        print("⚠️ Warning: Ollama might not be installed or running.")
        print("   Please ensure Ollama is installed and running: ollama serve")
        print("   You can still explore the code, but queries won't work without an LLM.\n")
    
    main()


