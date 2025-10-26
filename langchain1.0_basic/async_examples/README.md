# LangChain Async Examples

This directory contains advanced examples demonstrating asynchronous patterns with LangChain and Ollama.

## 📚 Overview

While synchronous LangChain code works fine for simple scripts, **async patterns unlock significant performance improvements** when working with:
- Multiple LLM calls
- Agent coordination
- Batch processing
- Concurrent user requests
- I/O-bound operations

## 🚀 Performance Benefits

Typical speedups with async:
- **2-4x faster** for concurrent operations
- **Linear scaling** with number of parallel tasks
- **Better resource utilization**
- **Lower latency** for web applications

## 📁 Files

### Core Async Agents

#### `async_planner_agent.py`
Enhanced async planner demonstrating:
- ✅ Concurrent plan generation
- ✅ Sequential vs concurrent comparison
- ✅ Streaming responses
- ✅ Multi-model parallel inference
- ✅ Batch processing with progress tracking

**Run:**
```bash
python async_examples/async_planner_agent.py
```

#### `async_research_agent.py`
Enhanced async researcher demonstrating:
- ✅ Concurrent research queries
- ✅ Multi-perspective analysis
- ✅ Rate-limited batch processing
- ✅ Streaming research results
- ✅ Progress tracking with `as_completed`

**Run:**
```bash
python async_examples/async_research_agent.py
```

### Demo Scripts

#### `concurrent_agents_demo.py`
Multi-agent coordination patterns:
- ✅ Sequential pipeline (baseline)
- ✅ Concurrent multiple queries
- ✅ Parallel multi-agent systems
- ✅ Optimized research→planning pipeline
- ✅ Error handling in concurrent operations

**Run:**
```bash
python async_examples/concurrent_agents_demo.py
```

#### `performance_comparison.py`
Direct performance comparison:
- ✅ Synchronous execution
- ✅ Async sequential execution
- ✅ Async concurrent execution
- ✅ Detailed timing metrics
- ✅ Speedup analysis

**Run:**
```bash
python async_examples/performance_comparison.py
```

## 🎯 When to Use Async

### ✅ Use Async When:
- Making multiple LLM calls
- Building web APIs (FastAPI, Starlette)
- Processing batches of inputs
- Coordinating multiple agents
- Handling concurrent user requests
- Doing I/O-bound work

### ❌ Stick with Sync When:
- Writing simple scripts
- Processing single queries
- Working in notebooks
- Sequential operations only
- No concurrency needed
- CPU-bound tasks

## 🔑 Key Async Patterns

### Pattern 1: Concurrent Execution
```python
# Instead of:
for query in queries:
    result = await llm.ainvoke(query)  # Wait for each

# Do this:
tasks = [llm.ainvoke(query) for query in queries]
results = await asyncio.gather(*tasks)  # All at once!
```

### Pattern 2: Streaming
```python
async for chunk in llm.astream(messages):
    print(chunk.content, end="", flush=True)
```

### Pattern 3: Progress Tracking
```python
tasks = [process(item) for item in items]
for completed_task in asyncio.as_completed(tasks):
    result = await completed_task
    print(f"Completed: {result}")
```

### Pattern 4: Rate Limiting
```python
for i in range(0, len(queries), batch_size):
    batch = queries[i:i + batch_size]
    results = await asyncio.gather(*[process(q) for q in batch])
    await asyncio.sleep(delay)  # Rate limit
```

### Pattern 5: Error Handling
```python
# Don't let one failure stop all tasks
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        handle_error(result)
    else:
        process_success(result)
```

## 📊 Example Performance

Running 5 concurrent queries:

| Method | Time | Speedup |
|--------|------|---------|
| Synchronous | 25.0s | 1.0x (baseline) |
| Async Sequential | 25.2s | ~1.0x (minimal overhead) |
| **Async Concurrent** | **6.5s** | **3.8x faster!** ⚡ |

*Note: Actual times depend on model, hardware, and query complexity*

## 🛠️ Technical Details

### LangChain Async Methods

Every LangChain synchronous method has an async equivalent:

| Sync | Async | Use Case |
|------|-------|----------|
| `invoke()` | `ainvoke()` | Single call |
| `batch()` | `abatch()` | Multiple calls |
| `stream()` | `astream()` | Token streaming |

### asyncio Patterns Used

- `asyncio.gather()` - Run multiple coroutines concurrently
- `asyncio.as_completed()` - Process results as they complete
- `asyncio.create_task()` - Start background tasks
- `asyncio.sleep()` - Non-blocking delays
- `return_exceptions=True` - Handle errors gracefully

## 💡 Best Practices

1. **Use `ainvoke` for all LLM calls in async code**
   ```python
   # Good
   response = await llm.ainvoke(messages)
   
   # Bad (blocks event loop)
   response = llm.invoke(messages)
   ```

2. **Gather related operations**
   ```python
   # Efficient
   results = await asyncio.gather(
       research_task(),
       planning_task(),
       analysis_task()
   )
   ```

3. **Handle errors with return_exceptions**
   ```python
   results = await asyncio.gather(*tasks, return_exceptions=True)
   ```

4. **Use rate limiting for APIs with limits**
   ```python
   semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
   async with semaphore:
       result = await api_call()
   ```

5. **Prefer `as_completed` for progress tracking**
   ```python
   for coro in asyncio.as_completed(tasks):
       result = await coro
       update_progress()
   ```

## 🎓 Learning Path

1. **Start**: Run `performance_comparison.py` to see the speedup
2. **Learn**: Read `async_planner_agent.py` for patterns
3. **Practice**: Run `concurrent_agents_demo.py` for real coordination
4. **Experiment**: Modify examples for your use cases

## 🔗 Related Documentation

- [LangChain Async Docs](https://python.langchain.com/docs/guides/async)
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)
- [FastAPI (async web framework)](https://fastapi.tiangolo.com/)

## 🤔 Common Questions

**Q: Does async make everything faster?**  
A: Only I/O-bound operations (like API calls, database queries, LLM calls). CPU-bound work won't benefit.

**Q: Can I mix sync and async code?**  
A: Yes, but be careful. Use `asyncio.run()` to call async from sync, and `await` for sync from async.

**Q: When should I NOT use async?**  
A: Simple scripts, notebooks, sequential processing, or when you're not doing multiple I/O operations.

**Q: Does async use more memory?**  
A: Slightly, but the performance gains usually far outweigh the memory cost.

## 🚦 Next Steps

- Try modifying the examples with your own queries
- Build a concurrent web API with FastAPI
- Experiment with different concurrency levels
- Add async to your existing agents

## 📝 Notes

- Ollama must be running locally
- Models must be pulled (`ollama pull llama3.1`)
- Some models may not support structured output
- Adjust `temperature` for different use cases

---

**Happy async coding!** ⚡🚀
