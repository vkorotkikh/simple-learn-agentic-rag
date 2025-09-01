# 🤖 Agentic RAG System

A simplified, easy to prototype test Retrieval-Augmented Generation (RAG) system using multiple specialized agents powered by LangChain and Pydantic. This implementation features recursive and semantic chunking, document reranking, and comprehensive RAG evaluation using RAGAS.

This testbed systems serves more as a learning and experimental demonstrator of how AI Agents can be utilized to enhance RAG pipelines.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
  - Query Router Agent: Analyzes and classifies queries
  - Retriever Agent: Performs document retrieval
  - Reranker Agent: Reranks documents for relevance
  - Synthesizer Agent: Generates comprehensive answers
  - Evaluator Agent: Assesses response quality

- **Advanced Chunking Strategies**:
  - Recursive text splitting
  - Semantic chunking
  - Hybrid approach combining both methods

- **Document Reranking**: Uses cross-encoder models for improved relevance

- **RAG Evaluation**: Built-in evaluation using RAGAS metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
  - Answer Correctness

- **Flexible Configuration**: Comprehensive configuration system using Pydantic models

- **Multi-format Document Support**: Load documents from various formats:
  - Text files (.txt)
  - PDFs (.pdf)
  - Markdown (.md)
  - CSV files (.csv)
  - JSON files (.json)
  - HTML files (.html)

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama installed locally (for LLM inference) or OpenAI API key
- CUDA-capable GPU (optional, for faster processing)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd /home/nomad/Documents/RAG-Projects/simple-rag-learn/agentic-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Ollama (if using local models)

```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., llama2)
ollama pull llama2
```

### 3. Basic Usage

```python
from agentic_rag import AgenticRAG, ChunkingStrategy

# Initialize the system
rag_system = AgenticRAG(
    llm_model="llama2",  # or any Ollama model
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunking_strategy=ChunkingStrategy.HYBRID
)

# Add documents
documents = [
    "Your document text here...",
    "Another document..."
]
rag_system.add_documents(documents)

# Process a query
response = rag_system.process_query("What is deep learning?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.2%}")
```

### 4. Running the Example

```bash
python agentic_rag.py
```

This will run the interactive demo with sample documents about AI, RAG systems, and Python libraries.

## 🔧 Configuration

### Using Configuration Files

Create a configuration file (`config.yaml`):

```yaml
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  cache_folder: "./model_cache"

llm:
  provider: "ollama"
  model_name: "llama2"
  temperature: 0.3
  max_tokens: 2048

vector_store:
  type: "faiss"
  persist_directory: "./vectorstore"

chunking:
  strategy: "hybrid"
  chunk_size: 500
  chunk_overlap: 100

reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5

num_agents: 4
agent_roles:
  - "query_router"
  - "retriever"
  - "reranker"
  - "synthesizer"

enable_evaluation: true
```

Load configuration:

```python
from config_utils import RAGConfig, setup_rag_system

# Load configuration
config = RAGConfig.load("config.yaml")

# Setup system with configuration
system_components = setup_rag_system("config.yaml")
```

### Generate Default Configurations

```bash
python config_utils.py
```

This creates:
- `config.yaml`: Default configuration
- `config_advanced.yaml`: Advanced configuration with optimized settings

## 📚 Loading Documents

### From Directory

```python
from config_utils import DocumentLoader

# Load all documents from a directory
loader = DocumentLoader()
documents = loader.load_directory(
    "/path/to/documents",
    glob_pattern="**/*.txt",  # Pattern for file selection
    recursive=True
)

# Add to RAG system
rag_system.add_documents(
    [doc.page_content for doc in documents],
    [doc.metadata for doc in documents]
)
```

### From Various Formats

```python
# Load PDF
pdf_docs = DocumentLoader.load_document("document.pdf")

# Load CSV
csv_docs = DocumentLoader.load_document("data.csv")

# Load from DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
docs = DocumentLoader.load_from_dataframe(
    df, 
    content_column="text",
    metadata_columns=["source", "date"]
)
```

## 🎯 Advanced Usage

### Custom Agent Configuration

```python
# Create system with custom agent configuration
rag_system = AgenticRAG(
    llm_model="mixtral",
    embedding_model="BAAI/bge-large-en-v1.5",
    chunking_strategy=ChunkingStrategy.SEMANTIC
)

# Process query with async support
import asyncio
response = asyncio.run(rag_system.process_query_async("Your query here"))
```

### Query Preprocessing

```python
from config_utils import QueryPreprocessor

# Clean and expand queries
preprocessor = QueryPreprocessor()
clean_query = preprocessor.clean_query("  What is   RAG???  ")
expanded_queries = preprocessor.expand_query("deep learning", num_expansions=3)
```

### Response Caching

```python
from config_utils import ResponseCache

# Initialize cache
cache = ResponseCache(cache_dir="./cache", ttl=3600)

# Check cache before processing
cached_response = cache.get(query)
if cached_response:
    return cached_response

# Process and cache
response = rag_system.process_query(query)
cache.set(query, response.dict())
```

### Performance Monitoring

```python
from config_utils import PerformanceMonitor
import time

monitor = PerformanceMonitor()

# Record query metrics
start_time = time.time()
response = rag_system.process_query(query)
response_time = time.time() - start_time

monitor.record_query(response_time, response.confidence)
print(monitor.get_summary())
```

## 📊 Evaluation

### Running Evaluation

```python
from config_utils import EvaluationDataset

# Create evaluation dataset
eval_dataset = EvaluationDataset()
eval_dataset.add_example(
    question="What is machine learning?",
    ground_truth="Machine learning is a subset of AI that enables systems to learn from data.",
    context=["Machine learning is...", "AI encompasses..."]
)

# Save/load evaluation dataset
eval_dataset.to_json("eval_data.json")
eval_dataset.from_json("eval_data.json")

# Evaluate responses
for question, ground_truth in zip(eval_dataset.questions, eval_dataset.ground_truths):
    response = rag_system.process_query(question)
    # Evaluation metrics are automatically calculated and included in response
    print(f"Question: {question}")
    print(f"Metrics: {response.evaluation_metrics}")
```

## 🏗️ Architecture

📊 **Visual Workflow Diagrams**: See [workflow_diagrams.md](workflow_diagrams.md) for comprehensive visual representations of:
- Complete System Architecture
- Agent Interaction Sequence
- Chunking Strategies Workflow
- High-Level System Overview

### Simple Architecture Overview:

```
┌─────────────────────────────────────────────────┐
│                  User Query                      │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│           Query Router Agent                     │
│         (Intent Classification)                  │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│           Retriever Agent                        │
│    (Document Retrieval with Chunking)           │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│           Reranker Agent                         │
│       (Cross-encoder Reranking)                 │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│          Synthesizer Agent                       │
│         (Answer Generation)                      │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│              RAG Evaluator                       │
│         (RAGAS Metrics Calculation)              │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│              Final Response                      │
│   (Answer + Confidence + Metrics + Trace)       │
└──────────────────────────────────────────────────┘
```

## 🛠️ Customization

### Adding New Agents

```python
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Custom tool description"
    
    def _run(self, input: str):
        # Your custom logic here
        return "result"

# Add to agent system
custom_tool = CustomTool()
# Integrate with AgenticRAG class
```

### Custom Chunking Strategy

```python
from langchain.text_splitter import TextSplitter

class CustomChunker(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        # Your custom chunking logic
        return chunks

# Use in AdvancedChunker class
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Ensure Ollama is running
   ollama serve
   ```

2. **Memory Issues with Large Documents**
   - Reduce chunk_size in configuration
   - Use batch processing for large document sets

3. **Slow Performance**
   - Enable GPU support in reranker configuration
   - Use smaller embedding models
   - Enable caching

4. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt --upgrade
   ```

## 📈 Performance Optimization

1. **Use GPU acceleration**:
   ```python
   config = RAGConfig(
       reranker=RerankerConfig(use_gpu=True)
   )
   ```

2. **Enable caching**:
   ```python
   config = RAGConfig(cache_enabled=True)
   ```

3. **Optimize chunk size**:
   - Larger chunks: Better context, slower processing
   - Smaller chunks: Faster processing, might lose context

4. **Use appropriate models**:
   - Lighter models for development/testing
   - Heavier models for production

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- LangChain for the agent framework
- RAGAS for evaluation metrics
- Hugging Face for embedding models
- Ollama for local LLM inference

## 📞 Support

For issues, questions, or suggestions, please open an issue in the project repository.

---

**Happy RAG-ing! 🚀**
