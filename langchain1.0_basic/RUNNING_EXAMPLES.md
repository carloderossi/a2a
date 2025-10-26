# Quick Reference: Running Examples

## Using `uv run` (Recommended)

The fastest way to run examples without activating the virtual environment:

```bash
# Basic examples
uv run examples/01_basic_ollama.py
uv run examples/02_tools_structured.py
uv run examples/03_mcp_integration.py

# Async examples
uv run async_examples/quick_start.py
uv run async_examples/performance_comparison.py
uv run async_examples/async_planner_agent.py
uv run async_examples/async_research_agent.py
uv run async_examples/concurrent_agents_demo.py

# Agents (requires A2A setup)
uv run agents/langchain_research_agent.py
uv run agents/langchain_planner_agent.py
```

## Using Traditional `python` Command

If you prefer the traditional approach:

```bash
# Activate virtual environment first
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Then run examples
python examples/01_basic_ollama.py
python async_examples/quick_start.py
# ... etc
```

## Quick Commands

```bash
# Install dependencies
uv pip install -e .

# Run any example
uv run <path/to/script.py>

# Check what's installed
uv pip list

# Update dependencies
uv pip install --upgrade langchain langchain-core langchain-ollama
```

## Troubleshooting

### If `uv run` doesn't work:
```bash
# Make sure uv is installed
pip install uv

# Recreate virtual environment
uv venv
uv pip install -e .
```

### If you get import errors:
```bash
# Reinstall dependencies
uv pip install -e .
```

### If Ollama connection fails:
```bash
# Check Ollama is running
ollama list

# Pull required models
ollama pull llama3.1
ollama pull mistral
ollama pull llama3.2
```

## Performance Comparison

| Method | Speed | Activation Needed | Notes |
|--------|-------|-------------------|-------|
| `uv run` | ⚡ Fast | ❌ No | Recommended |
| `python` | 🐌 Slower | ✅ Yes | Traditional |

## Why Use `uv run`?

1. **No activation** - Skip the `.venv\Scripts\activate` step
2. **Faster** - Written in Rust, much faster than pip
3. **Convenient** - Just `uv run` and go
4. **Safe** - Always uses the correct environment

## Examples

```bash
# Quick start with async
uv run async_examples/quick_start.py

# Performance comparison
uv run async_examples/performance_comparison.py

# Run multiple agents (in separate terminals)
uv run agents/langchain_research_agent.py
uv run agents/langchain_planner_agent.py
```

---

**Pro tip:** Add `uv` to your PATH and you can use it from anywhere in your project!
