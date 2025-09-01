"""
LangChain Agents Explained: AgentExecutor and create_react_agent

This file provides comprehensive examples and explanations of how AgentExecutor 
and create_react_agent work in LangChain, with practical examples.
"""

import os
from typing import List, Dict, Any, Optional, Type
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, tool
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

# ==================== CONCEPT EXPLANATION ====================

"""
🤖 WHAT IS create_react_agent?

create_react_agent is a LangChain function that creates a ReAct (Reasoning + Acting) agent.
ReAct agents follow this pattern:

1. **Thought**: The agent thinks about what to do
2. **Action**: The agent decides which tool to use
3. **Action Input**: The agent provides input to the tool
4. **Observation**: The agent observes the tool's output
5. **Thought**: The agent thinks about the observation
6. **Final Answer**: The agent provides the final response

Key components:
- **LLM**: The language model that powers the agent's reasoning
- **Tools**: Available actions the agent can take
- **Prompt**: Template that guides the agent's behavior

🔧 WHAT IS AgentExecutor?

AgentExecutor is the runtime engine that executes the agent created by create_react_agent.
It handles:
- Managing the conversation loop
- Parsing agent decisions
- Executing tool calls
- Error handling
- Memory management
- Verbose logging

Think of it as:
- create_react_agent = The agent's "brain" (decision-making logic)
- AgentExecutor = The agent's "body" (execution engine)
"""

# ==================== SIMPLE EXAMPLE ====================

class CalculatorTool(BaseTool):
    """Simple calculator tool for demonstration"""
    name = "calculator"
    description = "Useful for doing math calculations. Input should be a mathematical expression."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute the calculator"""
        try:
            # Safe evaluation of mathematical expressions
            result = eval(query)
            return f"The answer is {result}"
        except Exception as e:
            return f"Error in calculation: {e}"

class SearchTool(BaseTool):
    """Mock search tool for demonstration"""
    name = "search"
    description = "Useful for searching for information. Input should be a search query."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Mock search function"""
        # This is a mock - in reality, this would search the web or a database
        mock_results = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "langchain": "LangChain is a framework for developing applications powered by language models.",
            "machine learning": "Machine learning is a subset of AI that enables computers to learn from data.",
            "default": "I found some general information about your query."
        }
        
        query_lower = query.lower()
        for key, value in mock_results.items():
            if key in query_lower:
                return value
        return mock_results["default"]

def simple_agent_example():
    """Demonstrates basic AgentExecutor and create_react_agent usage"""
    print("="*80)
    print("🔧 SIMPLE AGENT EXAMPLE")
    print("="*80)
    
    # 1. Initialize the LLM
    llm = Ollama(model="llama2", temperature=0.1)
    
    # 2. Create tools
    tools = [CalculatorTool(), SearchTool()]
    
    # 3. Create the prompt template
    prompt = PromptTemplate(
        template="""You are a helpful assistant that can use tools to answer questions.

Available tools:
{tools}

Tool names: {tool_names}

When you need to use a tool, use this format:
Thought: I need to figure out...
Action: tool_name
Action Input: input_for_tool
Observation: tool_output

You can use multiple tools if needed. When you have enough information, provide your final answer.

Question: {input}
Thought: {agent_scratchpad}""",
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )
    
    # 4. Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # 5. Create the AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Shows the agent's reasoning process
        handle_parsing_errors=True,  # Gracefully handles LLM parsing errors
        max_iterations=5,  # Prevents infinite loops
        return_intermediate_steps=True  # Returns the thinking process
    )
    
    # 6. Test the agent
    test_queries = [
        "What is 25 * 4 + 10?",
        "Search for information about Python programming",
        "Calculate 100 / 5 and then search for information about the result"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*60}")
        print(f"Query {i}: {query}")
        print(f"{'─'*60}")
        
        try:
            result = agent_executor.run(query)
            print(f"\n✅ Final Answer: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

# ==================== ADVANCED EXAMPLE ====================

class DatabaseQueryTool(BaseTool):
    """Mock database query tool"""
    name = "database_query"
    description = "Query a database for information. Input should be a SQL-like query description."
    
    def __init__(self):
        super().__init__()
        # Mock database
        self.database = {
            "users": [
                {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
                {"id": 2, "name": "Bob", "age": 25, "city": "San Francisco"},
                {"id": 3, "name": "Charlie", "age": 35, "city": "New York"}
            ],
            "products": [
                {"id": 1, "name": "Laptop", "price": 1000, "category": "Electronics"},
                {"id": 2, "name": "Book", "price": 20, "category": "Education"},
                {"id": 3, "name": "Phone", "price": 800, "category": "Electronics"}
            ]
        }
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute database query"""
        query_lower = query.lower()
        
        if "users" in query_lower:
            if "new york" in query_lower:
                ny_users = [u for u in self.database["users"] if u["city"] == "New York"]
                return f"Users in New York: {ny_users}"
            else:
                return f"All users: {self.database['users']}"
        
        elif "products" in query_lower:
            if "electronics" in query_lower:
                electronics = [p for p in self.database["products"] if p["category"] == "Electronics"]
                return f"Electronics products: {electronics}"
            else:
                return f"All products: {self.database['products']}"
        
        return "Query not understood. Try asking about users or products."

class AnalyticsTool(BaseTool):
    """Mock analytics tool"""
    name = "analytics"
    description = "Perform analytics on data. Input should describe the analysis needed."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Perform analytics"""
        query_lower = query.lower()
        
        if "average" in query_lower and "price" in query_lower:
            # Mock calculation
            return "Average product price: $606.67"
        elif "count" in query_lower and "users" in query_lower:
            return "Total users: 3"
        elif "oldest" in query_lower and "user" in query_lower:
            return "Oldest user: Charlie (age 35)"
        else:
            return "Analytics query processed. Results would depend on specific data."

def advanced_agent_example():
    """Demonstrates advanced agent usage with multiple tools and complex reasoning"""
    print("\n" + "="*80)
    print("🚀 ADVANCED AGENT EXAMPLE")
    print("="*80)
    
    # Initialize LLM
    llm = Ollama(model="llama2", temperature=0.1)
    
    # Create advanced tools
    tools = [
        DatabaseQueryTool(),
        AnalyticsTool(),
        CalculatorTool()
    ]
    
    # Advanced prompt template with more specific instructions
    prompt = PromptTemplate(
        template="""You are a data analyst assistant with access to database and analytics tools.

Your goal is to help users analyze data by:
1. Querying databases for raw data
2. Performing calculations and analytics
3. Providing insights and summaries

Available tools:
{tools}

Tool names: {tool_names}

Always think step by step:
1. Understand what the user is asking
2. Determine what data you need
3. Use appropriate tools to gather data
4. Analyze the results
5. Provide a comprehensive answer

Use this format for tool usage:
Thought: I need to understand what the user wants...
Action: tool_name
Action Input: specific_input
Observation: tool_result

Continue this process until you have enough information to answer completely.

Question: {input}
{agent_scratchpad}""",
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )
    
    # Create agent with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create executor with advanced configuration
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True,
        early_stopping_method="generate"  # Stop when final answer is generated
    )
    
    # Complex test queries
    complex_queries = [
        "How many users do we have in our database?",
        "What's the average price of our products?",
        "Show me all users from New York and calculate how many there are",
        "Compare the number of electronics products to the total number of products"
    ]
    
    for i, query in enumerate(complex_queries, 1):
        print(f"\n{'─'*60}")
        print(f"Complex Query {i}: {query}")
        print(f"{'─'*60}")
        
        try:
            result = agent_executor.run(query)
            print(f"\n✅ Final Answer: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

# ==================== REACT PATTERN EXPLANATION ====================

def explain_react_pattern():
    """Explains the ReAct pattern in detail"""
    print("\n" + "="*80)
    print("🧠 UNDERSTANDING THE REACT PATTERN")
    print("="*80)
    
    react_explanation = """
The ReAct (Reasoning + Acting) pattern combines reasoning traces and task-specific actions.

📋 REACT PATTERN FLOW:

1. **Thought**: Agent analyzes the problem
   - "I need to find information about X"
   - "The user is asking for Y, so I should..."

2. **Action**: Agent selects a tool to use
   - Action: search
   - Action: calculator
   - Action: database_query

3. **Action Input**: Agent provides input to the tool
   - Action Input: "Python programming language"
   - Action Input: "25 * 4 + 10"

4. **Observation**: Agent receives tool output
   - Observation: "Python is a high-level programming language..."
   - Observation: "The answer is 110"

5. **Thought**: Agent reasons about the observation
   - "Now I have the calculation result..."
   - "Based on this information, I can conclude..."

6. **Final Answer**: Agent provides the complete response
   - "Based on my search and calculations, the answer is..."

🔄 ITERATION:
The agent can repeat steps 1-5 multiple times, using different tools and building 
up information until it has enough to provide a final answer.

🛡️ ERROR HANDLING:
- If a tool fails, the agent can try a different approach
- If the LLM generates invalid format, AgentExecutor handles parsing errors
- Max iterations prevent infinite loops

💡 BENEFITS:
- Transparent reasoning process
- Tool usage is deliberate and explained
- Can combine multiple tools in sequence
- Handles complex multi-step problems
"""
    
    print(react_explanation)

# ==================== PRACTICAL TIPS ====================

def practical_tips():
    """Provides practical tips for using AgentExecutor and create_react_agent"""
    print("\n" + "="*80)
    print("💡 PRACTICAL TIPS")
    print("="*80)
    
    tips = """
🔧 CONFIGURATION BEST PRACTICES:

1. **Prompt Design**:
   - Be specific about tool usage format
   - Include examples in the prompt
   - Clearly define the agent's role
   - Specify when to stop reasoning

2. **AgentExecutor Settings**:
   - verbose=True: See the agent's thinking (great for debugging)
   - handle_parsing_errors=True: Graceful error handling
   - max_iterations: Prevent infinite loops (usually 5-15)
   - return_intermediate_steps: Get the full reasoning chain

3. **Tool Design**:
   - Clear, descriptive names
   - Good descriptions explaining when to use the tool
   - Consistent input/output formats
   - Error handling within tools

4. **Memory Management**:
   - Use ConversationBufferMemory for context
   - Consider ConversationSummaryMemory for long conversations
   - Clear memory when switching contexts

⚠️ COMMON PITFALLS:

1. **Prompt Too Vague**: Agent doesn't know when to use tools
2. **No Max Iterations**: Agent gets stuck in loops
3. **Poor Tool Descriptions**: Agent uses wrong tools
4. **No Error Handling**: Crashes on unexpected inputs
5. **Verbose=False in Development**: Can't debug issues

🚀 PERFORMANCE OPTIMIZATION:

1. **Use faster LLMs for simple tasks**
2. **Cache tool results when possible**
3. **Limit tool descriptions to essential information**
4. **Use early_stopping_method="generate"**
5. **Consider async execution for multiple agents**

📊 MONITORING:

1. **Track tool usage frequency**
2. **Monitor reasoning chain length**
3. **Log parsing errors**
4. **Measure response times**
5. **Analyze success/failure rates**
"""
    
    print(tips)

# ==================== COMPARISON WITH SIMPLE RAG ====================

def compare_with_simple_rag():
    """Compares agent-based RAG with simple RAG"""
    print("\n" + "="*80)
    print("⚖️ AGENTS vs SIMPLE RAG")
    print("="*80)
    
    comparison = """
📊 SIMPLE RAG:
┌─────────────────────────────────────────────────────────────┐
│ Query → Embedding → Vector Search → Rerank → LLM → Answer   │
└─────────────────────────────────────────────────────────────┘

✅ Pros:
- Fast and predictable
- Simple to implement
- Low latency
- Deterministic flow

❌ Cons:
- Fixed pipeline
- Can't adapt to query complexity
- Limited reasoning capability
- Single retrieval strategy

🤖 AGENTIC RAG:
┌─────────────────────────────────────────────────────────────┐
│ Query → Agent Router → Agent Retriever → Agent Reranker     │
│    ↓                                                        │
│ Agent Synthesizer ← Agent Evaluator ← Multi-tool Usage     │
└─────────────────────────────────────────────────────────────┘

✅ Pros:
- Adaptive to query type
- Multi-step reasoning
- Can use multiple strategies
- Transparent decision making
- Error recovery
- Quality assurance

❌ Cons:
- Higher latency
- More complex to debug
- Requires more computational resources
- Potential for loops/errors

🎯 WHEN TO USE EACH:

Simple RAG:
- High-volume, similar queries
- Performance-critical applications
- Well-defined use cases
- Limited computational resources

Agentic RAG:
- Complex, varied queries
- Need for transparency
- Research and analysis tasks
- Quality is more important than speed
- Multi-step reasoning required
"""
    
    print(comparison)

# ==================== MAIN DEMONSTRATION ====================

def main():
    """Main function demonstrating all concepts"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     🤖 LANGCHAIN AGENTS: AgentExecutor & create_react_agent  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📚 This demonstration covers:")
    print("1. Basic concepts and ReAct pattern")
    print("2. Simple agent example")
    print("3. Advanced multi-tool agent")
    print("4. Practical tips and best practices")
    print("5. Comparison with simple RAG")
    
    try:
        # Explain concepts
        explain_react_pattern()
        
        # Simple example
        print("\n" + "🔍 Running simple example...")
        print("(Note: Requires Ollama to be running)")
        # simple_agent_example()  # Commented out to avoid requiring Ollama
        
        # Advanced example
        print("\n" + "🔍 Running advanced example...")
        print("(Note: Requires Ollama to be running)")
        # advanced_agent_example()  # Commented out to avoid requiring Ollama
        
        # Tips and comparison
        practical_tips()
        compare_with_simple_rag()
        
        print("\n" + "="*80)
        print("✨ SUMMARY")
        print("="*80)
        print("""
Key Takeaways:

1. **create_react_agent** creates the agent's reasoning logic
2. **AgentExecutor** provides the runtime environment  
3. **ReAct pattern** enables transparent multi-step reasoning
4. **Tools** give agents capabilities beyond text generation
5. **Proper configuration** is crucial for reliable behavior

To run the live examples, ensure Ollama is installed and running:
- ollama serve
- ollama pull llama2

Then uncomment the example function calls in main().
        """)
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("This is likely because Ollama is not running.")
        print("The conceptual explanations above are still valid!")

if __name__ == "__main__":
    main()
