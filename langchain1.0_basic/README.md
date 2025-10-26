# LangChain 1.0 with Ollama, MCP, and A2A Integration

This project demonstrates how to use LangChain 1.0 with:
- Different Ollama LLM models
- Tool calls and structured outputs
- MCP (Model Context Protocol) integration
- A2A (Agent-to-Agent) protocol integration
- **Asynchronous execution patterns** 🚀

## Prerequisites

1. **Ollama** installed and running
   - Download from: https://ollama.ai/
   - Pull models: `ollama pull llama3.1`, `ollama pull mistral`, `ollama pull llama3.2`

2. **Python 3.11+** installed

3. **uv** package manager (recommended)
   ```bash
   pip install uv
   ```

4. **MCP Server** running (from your existing setup)
   - Should be running on `http://127.0.0.1:8080/mcp`

## Installation

1. Navigate to the project directory:
   ```bash
   cd C:\Carlo\projects\a2a\langchain1.0_basic
   ```

2. Create virtual environment and install dependencies using uv:
   ```bash
   uv venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/Mac
   
   uv pip install -e .
   ```

   Or using pip:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```

## Project Structure

```
langchain1.0_basic/
├── pyproject.toml           # Project dependencies
├── README.md                # This file
├── examples/
│   ├── 01_basic_ollama.py           # Basic LangChain with different Ollama models
│   ├── 02_tools_structured.py       # Tool calls and structured outputs
│   ├── 03_mcp_integration.py        # MCP tool integration
│   ├── 04_a2a_agent.py             # A2A agent with LangChain
│   └── 05_complete_example.py       # Complete example combining everything
├── async_examples/                   # ⭐ NEW: Async patterns and performance
│   ├── README.md                    # Detailed async documentation
│   ├── quick_start.py               # Start here for async basics
│   ├── async_planner_agent.py       # Advanced async planner patterns
│   ├── async_research_agent.py      # Advanced async research patterns
│   ├── concurrent_agents_demo.py    # Multi-agent coordination
│   └── performance_comparison.py    # Sync vs async benchmarks
├── agents/
│   ├── langchain_research_agent.py  # Research agent using LangChain
│   └── langchain_planner_agent.py   # Planner agent using LangChain
├── utils/
│   ├── __init__.py
│   ├── mcp_tools.py                 # MCP tool wrapper for LangChain
│   └── a2a_utils.py                 # A2A utilities for LangChain
└── client_langchain.py              # Client that uses LangChain agents via A2A
```

## Quick Start

### For Basic LangChain Usage
```bash
python examples/01_basic_ollama.py
```

### For Async Patterns (Recommended!) ⚡
```bash
python async_examples/quick_start.py
```

See **[async_examples/README.md](async_examples/README.md)** for comprehensive async documentation.

## Running the Examples

### 1. Basic Ollama Models
```bash
python examples/01_basic_ollama.py
```
Demonstrates using different Ollama models (llama3.1, mistral, llama3.2) with LangChain.

### 2. Tools and Structured Outputs
```bash
python examples/02_tools_structured.py
```
Shows tool calling and structured output generation with LangChain.

### 3. MCP Integration
```bash
python examples/03_mcp_integration.py
```
Demonstrates consuming MCP tools in LangChain chains.

### 4. A2A Agents
Start the agents (in separate terminals):
```bash
# Terminal 1
python agents/langchain_research_agent.py

# Terminal 2
python agents/langchain_planner_agent.py

# Terminal 3
python client_langchain.py
```

### 5. Complete Example
```bash
python examples/05_complete_example.py
```
Combines all features: multiple models, tools, MCP, and A2A.

## 🚀 Async Examples (NEW!)

The `async_examples/` directory contains advanced examples showing **2-4x performance improvements** through asynchronous execution:

### Quick Start with Async
```bash
python async_examples/quick_start.py
```
Learn async basics in 5 examples - perfect starting point!

### Advanced Async Patterns
```bash
# Enhanced async planner with concurrent generation
python async_examples/async_planner_agent.py

# Enhanced async researcher with batch processing
python async_examples/async_research_agent.py

# Multi-agent coordination patterns
python async_examples/concurrent_agents_demo.py

# Performance benchmarks (sync vs async)
python async_examples/performance_comparison.py
```

### When to Use Async?

✅ **Use Async for:**
- Multiple LLM calls
- Web APIs (FastAPI, Starlette)
- Batch processing
- Multi-agent systems
- Concurrent user requests

❌ **Stick with Sync for:**
- Simple scripts
- Single queries
- Jupyter notebooks
- No concurrency

**See [async_examples/README.md](async_examples/README.md) for detailed patterns and best practices.**

## Key Features Demonstrated

### 1. Multiple Ollama Models
- Switching between different models (llama3.1, mistral, llama3.2)
- Model-specific parameters and configurations
- Comparing outputs from different models

### 2. Tool Calling
- Custom tool definitions
- Tool binding to LangChain agents
- Structured tool outputs

### 3. Structured Outputs
- Pydantic model validation
- JSON schema-based outputs
- Type-safe responses

### 4. MCP Integration
- Reading MCP resources (Agent Cards)
- Calling MCP tools from LangChain
- Converting MCP tools to LangChain tools

### 5. A2A Integration
- LangChain-based A2A agents
- Agent Card registration via MCP
- Multi-agent workflows
- Message passing between agents

### 6. Asynchronous Execution ⚡ (NEW!)
- Concurrent LLM calls (2-4x speedup)
- Streaming responses
- Batch processing with rate limiting
- Multi-agent coordination
- Progress tracking
- Error handling patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangChain Application                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Ollama     │    │  MCP Tools   │    │  A2A Agents  │  │
│  │   Models     │    │  Integration │    │  Integration │  │
│  │              │    │              │    │              │  │
│  │ • llama3.1   │◄───┤ • Resources  │◄───┤ • AgentCard  │  │
│  │ • mistral    │    │ • Tools      │    │ • ClientFactory│
│  │ • llama3.2   │    │ • Wrapper    │    │ • Messages   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ▲                    ▲                    ▲          │
│         └────────────────────┴────────────────────┘          │
│                    LangChain Framework                       │
│                   (Sync & Async Methods)                     │
└─────────────────────────────────────────────────────────────┘
```

## Performance Comparison

Example timing for 5 concurrent queries:

| Method | Time | Speedup |
|--------|------|---------|
| Synchronous | ~25s | 1.0x (baseline) |
| Async Sequential | ~25s | ~1.0x |
| **Async Concurrent** | **~6.5s** | **⚡ 3.8x faster!** |

*Run `python async_examples/performance_comparison.py` for live benchmarks*

## Troubleshooting

### Ollama Connection Issues
If you get connection errors:
```bash
# Check if Ollama is running
ollama list

# Pull required models
ollama pull llama3.1
ollama pull mistral
ollama pull llama3.2
```

### MCP Server Not Found
Make sure your MCP server is running on port 8080:
```bash
# Check the server URL in utils/a2a_utils.py
MCP_SERVER_URL = "http://127.0.0.1:8080/mcp"
```

### A2A Port Conflicts
If ports 9001/9002 are in use, modify the ports in:
- `agents/langchain_research_agent.py`
- `agents/langchain_planner_agent.py`

## Learning Path

1. **Start**: Basic examples (`examples/01_basic_ollama.py`)
2. **Learn**: Async basics (`async_examples/quick_start.py`)
3. **Practice**: Advanced async (`async_examples/async_planner_agent.py`)
4. **Master**: Multi-agent coordination (`async_examples/concurrent_agents_demo.py`)

## References

- **LangChain Documentation**: https://python.langchain.com/
- **LangChain Async Guide**: https://python.langchain.com/docs/guides/async
- **Ollama**: https://ollama.ai/
- **A2A Protocol**: https://a2a-protocol.org/
- **MCP**: https://github.com/modelcontextprotocol
- **Python asyncio**: https://docs.python.org/3/library/asyncio.html

## License

MIT License
