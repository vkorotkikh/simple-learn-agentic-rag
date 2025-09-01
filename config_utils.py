"""
Configuration and Utilities for Agentic RAG System
Provides configuration management, data loading utilities, and helper functions.
"""

import os
import json
import yaml
import pickle
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import hashlib

from pydantic import BaseModel, Field, validator
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader
)
from langchain.schema import Document
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==================== Configuration Models ====================

class EmbeddingConfig(BaseModel):
    """Configuration for embedding models"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"normalize_embeddings": True})
    cache_folder: Optional[str] = None
    
class LLMConfig(BaseModel):
    """Configuration for LLM models"""
    provider: str = Field(default="ollama", description="LLM provider: ollama, openai, huggingface")
    model_name: str = Field(default="llama2")
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048)
    top_p: Optional[float] = Field(default=0.9, ge=0, le=1)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
class VectorStoreConfig(BaseModel):
    """Configuration for vector stores"""
    type: str = Field(default="faiss", description="Vector store type: faiss, chroma, pinecone, weaviate")
    persist_directory: Optional[str] = Field(default="./vectorstore")
    collection_name: str = Field(default="agentic_rag")
    distance_metric: str = Field(default="cosine")
    index_params: Dict[str, Any] = Field(default_factory=dict)
    
class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""
    strategy: str = Field(default="hybrid", description="Chunking strategy: recursive, semantic, hybrid")
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=100, ge=0)
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", ".", "!", "?", ";", ":", " "])
    semantic_breakpoint_type: str = Field(default="percentile")
    semantic_breakpoint_amount: int = Field(default=80, ge=0, le=100)
    
class RerankerConfig(BaseModel):
    """Configuration for document reranking"""
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=32)
    use_gpu: bool = Field(default=False)
    
class RAGConfig(BaseModel):
    """Main configuration for the RAG system"""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    
    # Agent configurations
    num_agents: int = Field(default=4, ge=2, le=10, description="Number of agents in the system")
    agent_roles: List[str] = Field(
        default_factory=lambda: ["query_router", "retriever", "reranker", "synthesizer"]
    )
    use_memory: bool = Field(default=True, description="Use conversation memory")
    memory_window_size: int = Field(default=10, description="Number of turns to remember")
    
    # Evaluation settings
    enable_evaluation: bool = Field(default=True)
    evaluation_metrics: List[str] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevancy", "context_relevancy"]
    )
    
    # System settings
    verbose: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    cache_enabled: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    @validator("agent_roles")
    def validate_agent_roles(cls, v, values):
        """Ensure agent roles match num_agents"""
        if "num_agents" in values and len(v) != values["num_agents"]:
            # Adjust roles to match num_agents
            if len(v) < values["num_agents"]:
                # Add default roles
                default_roles = ["monitor", "validator", "optimizer", "coordinator"]
                for role in default_roles:
                    if len(v) < values["num_agents"] and role not in v:
                        v.append(role)
            else:
                # Trim roles
                v = v[:values["num_agents"]]
        return v
    
    def save(self, path: str):
        """Save configuration to file"""
        path_obj = Path(path)
        data = self.dict()
        
        if path_obj.suffix == ".json":
            with open(path_obj, "w") as f:
                json.dump(data, f, indent=2)
        elif path_obj.suffix in [".yaml", ".yml"]:
            with open(path_obj, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
    
    @classmethod
    def load(cls, path: str) -> "RAGConfig":
        """Load configuration from file"""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path_obj.suffix == ".json":
            with open(path_obj, "r") as f:
                data = json.load(f)
        elif path_obj.suffix in [".yaml", ".yml"]:
            with open(path_obj, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
        
        return cls(**data)

# ==================== Document Loading Utilities ====================

class DocumentLoader:
    """Unified document loader for various file formats"""
    
    SUPPORTED_EXTENSIONS = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader
    }
    
    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> List[Document]:
        """Load a single document"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            # Try to load as text
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return [Document(page_content=content, metadata={"source": str(path)})]
            except Exception as e:
                raise ValueError(f"Unsupported file format: {ext}. Error: {e}")
        
        loader_class = cls.SUPPORTED_EXTENSIONS[ext]
        loader = loader_class(str(path))
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = str(path)
            doc.metadata["file_type"] = ext
            doc.metadata["loaded_at"] = datetime.now().isoformat()
        
        return documents
    
    @classmethod
    def load_directory(cls, 
                      directory_path: Union[str, Path],
                      glob_pattern: str = "**/*",
                      recursive: bool = True,
                      show_progress: bool = True) -> List[Document]:
        """Load all documents from a directory"""
        path = Path(directory_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        # Find all files
        if recursive:
            files = list(path.glob(glob_pattern))
        else:
            files = list(path.glob(glob_pattern.replace("**/", "")))
        
        # Filter supported files
        supported_files = [
            f for f in files 
            if f.is_file() and (f.suffix.lower() in cls.SUPPORTED_EXTENSIONS or f.suffix == "")
        ]
        
        documents = []
        iterator = tqdm(supported_files, desc="Loading documents") if show_progress else supported_files
        
        for file_path in iterator:
            try:
                docs = cls.load_document(file_path)
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return documents
    
    @classmethod
    def load_from_dataframe(cls, df: pd.DataFrame, 
                           content_column: str,
                           metadata_columns: Optional[List[str]] = None) -> List[Document]:
        """Load documents from a pandas DataFrame"""
        documents = []
        
        for _, row in df.iterrows():
            content = row[content_column]
            metadata = {}
            
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        metadata[col] = row[col]
            
            documents.append(Document(page_content=str(content), metadata=metadata))
        
        return documents

# ==================== Caching Utilities ====================

class ResponseCache:
    """Simple response caching for RAG queries"""
    
    def __init__(self, cache_dir: str = "./cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Check TTL
        modified_time = cache_path.stat().st_mtime
        current_time = datetime.now().timestamp()
        
        if current_time - modified_time > self.ttl:
            # Cache expired
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, query: str, response: Dict[str, Any]):
        """Cache response"""
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(response, f)
        except Exception as e:
            print(f"Error caching response: {e}")
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

# ==================== Evaluation Utilities ====================

class EvaluationDataset:
    """Utility for creating and managing evaluation datasets"""
    
    def __init__(self):
        self.questions = []
        self.ground_truths = []
        self.contexts = []
    
    def add_example(self, question: str, ground_truth: str, context: List[str]):
        """Add an evaluation example"""
        self.questions.append(question)
        self.ground_truths.append(ground_truth)
        self.contexts.append(context)
    
    def from_json(self, file_path: str):
        """Load evaluation dataset from JSON"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            self.add_example(
                item["question"],
                item["ground_truth"],
                item.get("contexts", [])
            )
    
    def to_json(self, file_path: str):
        """Save evaluation dataset to JSON"""
        data = [
            {
                "question": q,
                "ground_truth": gt,
                "contexts": ctx
            }
            for q, gt, ctx in zip(self.questions, self.ground_truths, self.contexts)
        ]
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_dataset_dict(self) -> Dict[str, List]:
        """Get dataset as dictionary"""
        return {
            "question": self.questions,
            "ground_truth": self.ground_truths,
            "contexts": self.contexts
        }

# ==================== Query Processing Utilities ====================

class QueryPreprocessor:
    """Preprocess and enhance queries"""
    
    @staticmethod
    def expand_query(query: str, num_expansions: int = 3) -> List[str]:
        """Expand query with variations"""
        expansions = [query]
        
        # Add question variations
        if not query.endswith("?"):
            expansions.append(query + "?")
        
        # Add keyword extraction (simplified)
        keywords = [word for word in query.split() if len(word) > 3]
        if keywords:
            expansions.append(" ".join(keywords))
        
        # Add rephrasing templates
        rephrase_templates = [
            f"What is {query}",
            f"Explain {query}",
            f"Tell me about {query}"
        ]
        
        for template in rephrase_templates[:num_expansions-1]:
            if template not in expansions:
                expansions.append(template)
        
        return expansions[:num_expansions]
    
    @staticmethod
    def clean_query(query: str) -> str:
        """Clean and normalize query"""
        # Remove extra whitespace
        query = " ".join(query.split())
        
        # Remove special characters (keep alphanumeric and basic punctuation)
        import re
        query = re.sub(r'[^\w\s\?\.\,\-]', '', query)
        
        return query.strip()
    
    @staticmethod
    def detect_language(query: str) -> str:
        """Detect query language (simplified)"""
        # This is a very basic implementation
        # In production, use proper language detection libraries
        
        # Check for common non-English characters
        if any(ord(char) > 127 for char in query):
            return "non-english"
        
        return "english"

# ==================== Monitoring and Logging ====================

class PerformanceMonitor:
    """Monitor RAG system performance"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0,
            "average_confidence": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.response_times = []
        self.confidence_scores = []
    
    def record_query(self, response_time: float, confidence: float, success: bool = True):
        """Record query metrics"""
        self.metrics["total_queries"] += 1
        
        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1
        
        self.response_times.append(response_time)
        self.confidence_scores.append(confidence)
        
        # Update averages
        self.metrics["average_response_time"] = np.mean(self.response_times)
        self.metrics["average_confidence"] = np.mean(self.confidence_scores)
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics["cache_misses"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def get_summary(self) -> str:
        """Get performance summary"""
        cache_hit_rate = (
            self.metrics["cache_hits"] / 
            (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
            else 0
        )
        
        success_rate = (
            self.metrics["successful_queries"] / self.metrics["total_queries"]
            if self.metrics["total_queries"] > 0
            else 0
        )
        
        summary = f"""
Performance Summary:
====================
Total Queries: {self.metrics['total_queries']}
Success Rate: {success_rate:.2%}
Average Response Time: {self.metrics['average_response_time']:.2f}s
Average Confidence: {self.metrics['average_confidence']:.2%}
Cache Hit Rate: {cache_hit_rate:.2%}
"""
        return summary

# ==================== Example Configuration Files ====================

def create_default_config(output_path: str = "config.yaml"):
    """Create a default configuration file"""
    config = RAGConfig()
    config.save(output_path)
    print(f"Default configuration saved to {output_path}")
    return config

def create_advanced_config(output_path: str = "config_advanced.yaml"):
    """Create an advanced configuration with custom settings"""
    config = RAGConfig(
        embedding=EmbeddingConfig(
            model_name="BAAI/bge-large-en-v1.5",
            cache_folder="./model_cache"
        ),
        llm=LLMConfig(
            provider="ollama",
            model_name="mixtral",
            temperature=0.2,
            max_tokens=4096
        ),
        vector_store=VectorStoreConfig(
            type="chroma",
            persist_directory="./chroma_db",
            collection_name="advanced_rag"
        ),
        chunking=ChunkingConfig(
            strategy="hybrid",
            chunk_size=800,
            chunk_overlap=200
        ),
        reranker=RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            top_k=10,
            use_gpu=True
        ),
        num_agents=5,
        agent_roles=["query_router", "retriever", "reranker", "synthesizer", "validator"],
        enable_evaluation=True
    )
    config.save(output_path)
    print(f"Advanced configuration saved to {output_path}")
    return config

# ==================== Main Utility Function ====================

def setup_rag_system(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Setup RAG system with configuration"""
    if config_path:
        config = RAGConfig.load(config_path)
    else:
        config = RAGConfig()
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize cache if enabled
    cache = ResponseCache(cache_ttl=config.cache_ttl) if config.cache_enabled else None
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    return {
        "config": config,
        "cache": cache,
        "monitor": monitor,
        "document_loader": DocumentLoader,
        "query_preprocessor": QueryPreprocessor
    }

if __name__ == "__main__":
    # Create example configuration files
    create_default_config("config.yaml")
    create_advanced_config("config_advanced.yaml")
    
    # Example usage
    print("\nExample Usage:")
    print("-" * 50)
    
    # Load configuration
    config = RAGConfig.load("config.yaml")
    print(f"Loaded config with {config.num_agents} agents")
    
    # Setup system
    system = setup_rag_system("config.yaml")
    print(f"System setup complete with cache: {system['cache'] is not None}")
    
    # Load documents
    loader = DocumentLoader()
    print("\nDocument loader ready for various file formats")
    print(f"Supported formats: {list(DocumentLoader.SUPPORTED_EXTENSIONS.keys())}")
