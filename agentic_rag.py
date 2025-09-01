"""
Agentic RAG System with Multiple Specialized Agents
Uses LangChain for agent orchestration and includes recursive/semantic chunking,
reranking, and RAG evaluation with RAGAS.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

# Core dependencies
import numpy as np
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, BaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.memory import ConversationBufferMemory
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.config import Settings

# RAG Evaluation
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Pydantic Models ====================

class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class AgentRole(str, Enum):
    QUERY_ROUTER = "query_router"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    SYNTHESIZER = "synthesizer"
    EVALUATOR = "evaluator"

class QueryIntent(BaseModel):
    """Model for query classification and routing"""
    query: str
    intent_type: str = Field(description="Type of query: factual, analytical, comparative, etc.")
    complexity: int = Field(ge=1, le=5, description="Query complexity from 1-5")
    domains: List[str] = Field(description="Relevant knowledge domains")
    confidence: float = Field(ge=0, le=1, description="Confidence in classification")

class RetrievedContext(BaseModel):
    """Model for retrieved and reranked context"""
    documents: List[Document]
    scores: List[float]
    reranked: bool = False
    chunking_strategy: ChunkingStrategy
    metadata: Dict[str, Any] = {}

class RAGResponse(BaseModel):
    """Model for final RAG response"""
    query: str
    answer: str
    contexts: List[str]
    confidence: float
    agent_trace: List[Dict[str, Any]]
    evaluation_metrics: Optional[Dict[str, float]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# ==================== Chunking Strategies ====================

class AdvancedChunker:
    """Advanced chunking with recursive and semantic strategies"""
    
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len,
        )
        self.semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80
        )
    
    def chunk(self, text: str, strategy: ChunkingStrategy) -> List[Document]:
        """Apply chunking strategy to text"""
        if strategy == ChunkingStrategy.RECURSIVE:
            return self.recursive_splitter.create_documents([text])
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self.semantic_splitter.create_documents([text])
        elif strategy == ChunkingStrategy.HYBRID:
            # Combine both strategies
            recursive_chunks = self.recursive_splitter.create_documents([text])
            semantic_chunks = self.semantic_splitter.create_documents([text])
            
            # Merge and deduplicate
            all_chunks = recursive_chunks + semantic_chunks
            unique_chunks = []
            seen_content = set()
            
            for chunk in all_chunks:
                if chunk.page_content not in seen_content:
                    unique_chunks.append(chunk)
                    seen_content.add(chunk.page_content)
            
            return unique_chunks
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

# ==================== Reranking Module ====================

class DocumentReranker:
    """Reranks retrieved documents using cross-encoder"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross_encoder = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> Tuple[List[Document], List[float]]:
        """Rerank documents based on relevance to query"""
        if not documents:
            return [], []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        top_docs = [pair[0] for pair in doc_score_pairs[:top_k]]
        top_scores = [pair[1] for pair in doc_score_pairs[:top_k]]
        
        return top_docs, top_scores

# ==================== Specialized Agent Tools ====================

class QueryRouterTool(BaseTool):
    """Tool for query analysis and routing"""
    name = "query_router"
    description = "Analyzes query intent and routes to appropriate knowledge base"
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        """Analyze query and determine routing"""
        # Simple intent classification (in production, use a proper classifier)
        intent_keywords = {
            "factual": ["what", "who", "when", "where"],
            "analytical": ["why", "how", "explain", "analyze"],
            "comparative": ["compare", "versus", "difference", "better"],
            "procedural": ["steps", "process", "method", "procedure"]
        }
        
        intent_type = "general"
        for intent, keywords in intent_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                intent_type = intent
                break
        
        # Determine complexity (simple heuristic)
        complexity = min(5, max(1, len(query.split()) // 10 + 1))
        
        # Determine domains (simplified)
        domains = ["general"]
        if any(word in query.lower() for word in ["science", "research", "study"]):
            domains.append("scientific")
        if any(word in query.lower() for word in ["code", "programming", "software"]):
            domains.append("technical")
        
        return {
            "query": query,
            "intent_type": intent_type,
            "complexity": complexity,
            "domains": domains,
            "confidence": 0.85
        }

class RetrieverTool(BaseTool):
    """Tool for document retrieval with advanced chunking"""
    name = "retriever"
    description = "Retrieves relevant documents using specified chunking strategy"
    
    def __init__(self, vector_store, chunker, chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID):
        super().__init__()
        self.vector_store = vector_store
        self.chunker = chunker
        self.chunking_strategy = chunking_strategy
    
    def _run(self, query: str, k: int = 10, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        """Retrieve relevant documents"""
        # Retrieve documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Calculate similarity scores (simplified)
        scores = [0.9 - i * 0.05 for i in range(len(docs))]
        
        return {
            "documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
            "scores": scores,
            "chunking_strategy": self.chunking_strategy.value,
            "num_retrieved": len(docs)
        }

class RerankerTool(BaseTool):
    """Tool for document reranking"""
    name = "reranker"
    description = "Reranks documents based on relevance to query"
    
    def __init__(self, reranker: DocumentReranker):
        super().__init__()
        self.reranker = reranker
    
    def _run(self, query: str, documents: List[Dict], top_k: int = 5, 
             run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        """Rerank documents"""
        # Convert back to Document objects
        doc_objects = [Document(page_content=doc["content"], metadata=doc.get("metadata", {})) 
                       for doc in documents]
        
        # Rerank
        reranked_docs, scores = self.reranker.rerank(query, doc_objects, top_k)
        
        return {
            "documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in reranked_docs],
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
            "reranked": True
        }

class SynthesizerTool(BaseTool):
    """Tool for answer synthesis"""
    name = "synthesizer"
    description = "Synthesizes final answer from retrieved and reranked contexts"
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.synthesis_prompt = PromptTemplate(
            template="""Based on the following context, provide a comprehensive answer to the query.
            
Query: {query}

Context:
{context}

Instructions:
1. Provide a clear, accurate answer based solely on the given context
2. If the context doesn't contain sufficient information, state that clearly
3. Include relevant details and examples from the context
4. Maintain factual accuracy

Answer:""",
            input_variables=["query", "context"]
        )
    
    def _run(self, query: str, documents: List[Dict], 
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Synthesize answer from documents"""
        # Combine document contents
        context = "\n\n".join([f"[{i+1}] {doc['content']}" for i, doc in enumerate(documents)])
        
        # Generate answer
        chain = LLMChain(llm=self.llm, prompt=self.synthesis_prompt)
        answer = chain.run(query=query, context=context)
        
        return answer

# ==================== RAG Evaluation ====================

class RAGEvaluator:
    """Evaluates RAG pipeline using RAGAS metrics"""
    
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_relevancy,
            answer_correctness
        ]
    
    def evaluate_response(self, query: str, answer: str, contexts: List[str], 
                          ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate a single RAG response"""
        # Prepare dataset for RAGAS
        eval_data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        
        if ground_truth:
            eval_data["ground_truth"] = [ground_truth]
        
        dataset = Dataset.from_dict(eval_data)
        
        try:
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics if ground_truth else self.metrics[:4],  # Skip answer_correctness without ground truth
            )
            
            return result.to_dict()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}

# ==================== Main Agentic RAG System ====================

class AgenticRAG:
    """Main Agentic RAG system orchestrating multiple specialized agents"""
    
    def __init__(self, 
                 documents_path: Optional[str] = None,
                 llm_model: str = "llama2",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID):
        
        # Initialize models
        self.llm = Ollama(model=llm_model, temperature=0.3)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize components
        self.chunker = AdvancedChunker(self.embeddings)
        self.reranker = DocumentReranker()
        self.evaluator = RAGEvaluator()
        
        # Initialize vector store
        self.vector_store = None
        if documents_path:
            self.load_documents(documents_path, chunking_strategy)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Conversation memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        logger.info("Agentic RAG system initialized successfully")
    
    def load_documents(self, path: str, chunking_strategy: ChunkingStrategy):
        """Load and index documents"""
        documents = []
        path_obj = Path(path)
        
        if path_obj.is_file():
            with open(path_obj, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = self.chunker.chunk(text, chunking_strategy)
                documents.extend(chunks)
        elif path_obj.is_dir():
            for file_path in path_obj.glob("**/*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    chunks = self.chunker.chunk(text, chunking_strategy)
                    for chunk in chunks:
                        chunk.metadata["source"] = str(file_path)
                    documents.extend(chunks)
        
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Loaded and indexed {len(documents)} document chunks")
        else:
            logger.warning("No documents found to load")
    
    def _initialize_agents(self) -> Dict[AgentRole, AgentExecutor]:
        """Initialize all specialized agents"""
        agents = {}
        
        # Query Router Agent
        router_tools = [QueryRouterTool()]
        router_prompt = PromptTemplate(
            template="""You are a query routing specialist. Analyze the user's query and determine:
            1. The intent type (factual, analytical, comparative, procedural)
            2. Query complexity (1-5 scale)
            3. Relevant knowledge domains
            
            {tools}
            {tool_names}
            
            Query: {input}
            {agent_scratchpad}
            
            Thought: Let me analyze this query...
            """,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )
        router_agent = create_react_agent(self.llm, router_tools, router_prompt)
        agents[AgentRole.QUERY_ROUTER] = AgentExecutor(
            agent=router_agent, 
            tools=router_tools, 
            verbose=True,
            handle_parsing_errors=True
        )

        # AgentExecutor is a runtime engine that executes the agents logic and tools.
        # It is now being deprecated in favor of LangGraph
        agents = AgentExecutor(
            agent=agent,        # the decision making logic
            tools=tools,        # tool execution logic
            verbose=True,       
            handle_parsing_errors=True, # Manages failures and parsing errors
            max_iterations=10, # Maximum number of iterations an agent can take; prevents infinite loops
            memory=self.memory  # Context retention
        )

        # Retriever Agent
        if self.vector_store:
            retriever_tools = [RetrieverTool(self.vector_store, self.chunker)]
            retriever_prompt = PromptTemplate(
                template="""You are a document retrieval specialist. Your job is to find the most relevant documents for the query.
                
                {tools}
                {tool_names}
                
                Query: {input}
                {agent_scratchpad}
                
                Thought: I need to retrieve relevant documents...
                """,
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
            )
            retriever_agent = create_react_agent(self.llm, retriever_tools, retriever_prompt)
            agents[AgentRole.RETRIEVER] = AgentExecutor(
                agent=retriever_agent,
                tools=retriever_tools,
                verbose=True,
                handle_parsing_errors=True
            )
        
        # Reranker Agent
        reranker_tools = [RerankerTool(self.reranker)]
        agents[AgentRole.RERANKER] = reranker_tools[0]  # Direct tool usage for reranking
        
        # Synthesizer Agent
        synthesizer_tools = [SynthesizerTool(self.llm)]
        agents[AgentRole.SYNTHESIZER] = synthesizer_tools[0]  # Direct tool usage for synthesis
        
        return agents
    
    async def process_query_async(self, query: str) -> RAGResponse:
        """Process query through the multi-agent pipeline asynchronously"""
        agent_trace = []
        
        try:
            # Step 1: Query Routing
            logger.info("Step 1: Analyzing query intent...")
            router_result = self.agents[AgentRole.QUERY_ROUTER].run(query)
            agent_trace.append({
                "agent": "query_router",
                "action": "analyze_query",
                "result": router_result
            })
            
            # Step 2: Document Retrieval
            if AgentRole.RETRIEVER in self.agents:
                logger.info("Step 2: Retrieving relevant documents...")
                retriever_result = self.agents[AgentRole.RETRIEVER].run(query)
                agent_trace.append({
                    "agent": "retriever",
                    "action": "retrieve_documents",
                    "result": retriever_result
                })
                
                # Parse retriever result (simplified - in production, properly parse the agent output)
                documents = []
                if isinstance(retriever_result, str):
                    # Extract documents from string output (simplified)
                    import re
                    doc_pattern = r'"content":\s*"([^"]*)"'
                    matches = re.findall(doc_pattern, retriever_result)
                    documents = [{"content": match} for match in matches[:10]]
                
                # Step 3: Reranking
                if documents:
                    logger.info("Step 3: Reranking documents...")
                    reranker_result = self.agents[AgentRole.RERANKER].run(
                        query=query,
                        documents=documents,
                        top_k=5
                    )
                    agent_trace.append({
                        "agent": "reranker",
                        "action": "rerank_documents",
                        "result": reranker_result
                    })
                    
                    # Use reranked documents
                    if reranker_result.get("documents"):
                        documents = reranker_result["documents"]
                
                # Step 4: Answer Synthesis
                logger.info("Step 4: Synthesizing answer...")
                answer = self.agents[AgentRole.SYNTHESIZER].run(
                    query=query,
                    documents=documents[:5]  # Use top 5 documents
                )
                agent_trace.append({
                    "agent": "synthesizer",
                    "action": "synthesize_answer",
                    "result": answer
                })
                
                # Extract contexts
                contexts = [doc["content"] for doc in documents[:5]]
                
                # Step 5: Evaluation (optional)
                logger.info("Step 5: Evaluating response...")
                eval_metrics = self.evaluator.evaluate_response(
                    query=query,
                    answer=answer,
                    contexts=contexts
                )
                
                return RAGResponse(
                    query=query,
                    answer=answer,
                    contexts=contexts,
                    confidence=0.85,
                    agent_trace=agent_trace,
                    evaluation_metrics=eval_metrics
                )
            else:
                # No vector store available
                return RAGResponse(
                    query=query,
                    answer="No documents available for retrieval. Please load documents first.",
                    contexts=[],
                    confidence=0.0,
                    agent_trace=agent_trace,
                    evaluation_metrics=None
                )
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return RAGResponse(
                query=query,
                answer=f"Error processing query: {str(e)}",
                contexts=[],
                confidence=0.0,
                agent_trace=agent_trace,
                evaluation_metrics=None
            )
    
    def process_query(self, query: str) -> RAGResponse:
        """Synchronous wrapper for query processing"""
        return asyncio.run(self.process_query_async(query))
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add new documents to the knowledge base"""
        documents = []
        for i, text in enumerate(texts):
            chunks = self.chunker.chunk(text, ChunkingStrategy.HYBRID)
            for chunk in chunks:
                if metadatas and i < len(metadatas):
                    chunk.metadata.update(metadatas[i])
                documents.append(chunk)
        
        if self.vector_store:
            self.vector_store.add_documents(documents)
        else:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        logger.info(f"Added {len(documents)} new document chunks")
        
        # Reinitialize agents with updated vector store
        self.agents = self._initialize_agents()

# ==================== Main Execution ====================

def main():
    """Main execution function with examples"""
    
    # Initialize the agentic RAG system
    print("🚀 Initializing Agentic RAG System...")
    rag_system = AgenticRAG(
        llm_model="llama2",  # Change to your preferred model
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunking_strategy=ChunkingStrategy.HYBRID
    )
    
    # Add sample documents
    print("\n📚 Adding sample documents to knowledge base...")
    sample_documents = [
        """Artificial Intelligence (AI) is transforming the world of technology. 
        Machine learning algorithms enable computers to learn from data without explicit programming.
        Deep learning, a subset of machine learning, uses neural networks with multiple layers.
        Natural language processing allows computers to understand and generate human language.
        Computer vision enables machines to interpret and understand visual information from the world.""",
        
        """RAG (Retrieval-Augmented Generation) combines retrieval systems with language models.
        It first retrieves relevant documents from a knowledge base, then uses them to generate responses.
        This approach reduces hallucinations and provides more accurate, grounded answers.
        Advanced RAG systems use techniques like reranking, query expansion, and hybrid search.
        Chunking strategies like recursive and semantic chunking improve retrieval quality.""",
        
        """Python is a versatile programming language widely used in data science and AI.
        Libraries like NumPy and Pandas provide powerful data manipulation capabilities.
        TensorFlow and PyTorch are popular frameworks for deep learning.
        Scikit-learn offers a wide range of machine learning algorithms.
        LangChain simplifies the development of applications powered by language models."""
    ]
    
    rag_system.add_documents(sample_documents)
    
    # Example queries
    example_queries = [
        "What is deep learning and how does it relate to AI?",
        "Explain how RAG systems work and their advantages",
        "What Python libraries are used for machine learning?",
        "Compare traditional machine learning with deep learning approaches"
    ]
    
    print("\n🔍 Processing example queries...")
    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")
        
        response = rag_system.process_query(query)
        
        print(f"\n📝 Answer:")
        print(response.answer)
        
        print(f"\n📊 Confidence: {response.confidence:.2%}")
        
        print(f"\n📚 Used {len(response.contexts)} context chunks")
        
        if response.evaluation_metrics:
            print(f"\n📈 Evaluation Metrics:")
            for metric, score in response.evaluation_metrics.items():
                if isinstance(score, (int, float)):
                    print(f"  - {metric}: {score:.3f}")
        
        print(f"\n🔄 Agent Trace ({len(response.agent_trace)} steps):")
        for step in response.agent_trace[:3]:  # Show first 3 steps
            print(f"  - {step['agent']}: {step['action']}")
    
    # Interactive mode
    print("\n" + "="*80)
    print("💬 Interactive Mode - Type 'quit' to exit")
    print("="*80)
    
    while True:
        user_query = input("\nEnter your query: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_query:
            response = rag_system.process_query(user_query)
            print(f"\n📝 Answer:")
            print(response.answer)
            print(f"\n📊 Confidence: {response.confidence:.2%}")
    
    print("\n👋 Thank you for using the Agentic RAG System!")

if __name__ == "__main__":
    main()
