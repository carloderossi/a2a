# LangChain 1.0 Best Practices Applied

## 🎯 Key Improvements Made

Based on feedback, the async agent examples have been completely refactored to follow LangChain 1.0 best practices.

---

## 1. ✅ Structured Output with `with_structured_output()`

### ❌ Before (Manual Parsing)
```python
response = await llm.ainvoke(messages)
return response.content  # Returns unstructured text
```

### ✅ After (Structured Output)
```python
# Define Pydantic model
class ActionPlan(BaseModel):
    objective: str = Field(description="Main objective")
    steps: List[str] = Field(description="Actionable steps")
    timeline: str = Field(description="Timeline estimate")
    resources: List[str] = Field(description="Required resources")
    criteria: List[str] = Field(description="Success criteria")

# Use structured output
structured_llm = self.llm.with_structured_output(ActionPlan)
plan: ActionPlan = await structured_llm.ainvoke(prompt)

# Now you get type-safe, validated objects!
print(plan.objective)  # Guaranteed to exist
print(plan.steps)      # Guaranteed to be a list
```

**Benefits:**
- ✅ Type safety at runtime
- ✅ Automatic validation with Pydantic
- ✅ IDE autocomplete support
- ✅ Consistent output format
- ✅ Easy to serialize/deserialize

---

## 2. ✅ LCEL Composition with `|` Operator

### ❌ Before (Manual Message Construction)
```python
messages = [
    SystemMessage(content="You are an expert..."),
    HumanMessage(content=f"Based on: {research}...")
]
response = await llm.ainvoke(messages)
```

### ✅ After (LCEL Chains)
```python
from langchain_core.prompts import ChatPromptTemplate

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert strategic planner."),
    ("human", "Based on this research:\n{research}\n\nCreate a plan.")
])

# Compose chain using | operator
chain = prompt | self.llm | ActionPlan

# Execute chain
result = await chain.ainvoke({"research": research})
```

**Benefits:**
- ✅ Declarative and composable
- ✅ Reusable components
- ✅ Easier to test
- ✅ Better separation of concerns
- ✅ Follows LangChain Expression Language (LCEL) standard

### Advanced LCEL Patterns
```python
# Chain with structured output
self.structured_llm = self.llm.with_structured_output(ActionPlan)
self.chain = self.prompt | self.structured_llm

# Chain with transformations
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"research": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

# Parallel chains
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    plan=planner_chain,
    research=research_chain
)
```

---

## 3. ✅ Comprehensive Error Handling

### ❌ Before (No Error Handling)
```python
async def generate_plan(self, research: str):
    response = await llm.ainvoke(messages)
    return response.content  # Crashes on error
```

### ✅ After (Robust Error Handling)
```python
async def generate_plan(self, research: str) -> Dict[str, Any]:
    start_time = datetime.now()
    error_msg = None
    
    try:
        logging.info(f"Generating plan...")
        
        # Use LCEL chain for execution
        plan: ActionPlan = await self.chain.ainvoke({"research": research})
        
        duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"✓ Plan generated in {duration:.2f}s")
        
        return {
            "plan": plan,
            "model": self.model_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        logging.error(f"✗ Plan generation failed: {error_msg}")
        
        return {
            "plan": None,
            "model": self.model_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "error": error_msg
        }
```

**Benefits:**
- ✅ Graceful degradation
- ✅ Detailed error information
- ✅ Performance metrics even on failure
- ✅ Proper logging
- ✅ No crashes in concurrent operations

### Error Handling in Concurrent Operations
```python
# Use return_exceptions=True to prevent one failure from stopping all
results = await asyncio.gather(*tasks, return_exceptions=True)

# Process results and exceptions separately
processed_results = []
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logging.error(f"✗ Task {i+1} failed: {result}")
        processed_results.append({"error": str(result)})
    else:
        processed_results.append(result)
```

---

## 4. 🎯 Additional Best Practices Applied

### A. Type Hints
```python
async def generate_plan(
    self, 
    research: str,
    return_metadata: bool = True
) -> Dict[str, Any] | ActionPlan:
    """Clear type signatures for better IDE support."""
```

### B. Logging
```python
import logging

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s %(message)s',
    level=logging.INFO
)

logging.info("Plan generated successfully")
logging.error("Plan generation failed", exc_info=True)
```

### C. Metadata Tracking
```python
return {
    "plan": plan,
    "model": self.model_name,
    "duration": duration,
    "timestamp": datetime.now().isoformat(),
    "error": None
}
```

### D. Progress Tracking
```python
tasks = [process(item) for item in items]

for i, task in enumerate(asyncio.as_completed(tasks), 1):
    result = await task
    print(f"✓ Completed {i}/{len(items)}")
```

---

## 📊 Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Output Format** | Unstructured text | Pydantic models |
| **Type Safety** | None | Full type hints |
| **Composition** | Manual messages | LCEL chains |
| **Error Handling** | None | Comprehensive |
| **Logging** | None | Structured logging |
| **Testability** | Difficult | Easy |
| **Reusability** | Low | High |
| **IDE Support** | Poor | Excellent |

---

## 🚀 Updated Files

### `async_planner_agent.py`
- ✅ Structured output with `ActionPlan` model
- ✅ LCEL composition: `prompt | llm | ActionPlan`
- ✅ Error handling in all async methods
- ✅ Progress tracking and logging
- ✅ Metadata tracking for all operations

### `async_research_agent.py`
- ✅ Structured output with `ResearchFindings` model
- ✅ LCEL composition: `prompt | llm`
- ✅ Error handling in all async methods
- ✅ Multi-perspective analysis with error handling
- ✅ Rate-limited batch processing with error handling

---

## 💡 Key Takeaways

### 1. Use Structured Output
```python
# Always prefer this:
structured_llm = llm.with_structured_output(MyModel)
result: MyModel = await structured_llm.ainvoke(input)

# Over this:
response = await llm.ainvoke(messages)
text = response.content  # Unstructured!
```

### 2. Use LCEL Chains
```python
# Always prefer this:
chain = prompt | llm | output_parser
result = await chain.ainvoke({"input": data})

# Over this:
messages = build_messages(data)
response = await llm.ainvoke(messages)
result = parse_response(response)
```

### 3. Always Handle Errors
```python
# Always do this:
try:
    result = await chain.ainvoke(input)
    return {"data": result, "error": None}
except Exception as e:
    logging.error(f"Failed: {e}")
    return {"data": None, "error": str(e)}

# Never do this:
result = await chain.ainvoke(input)  # Crashes on error
return result
```

---

## 🎓 Learning Resources

- **LCEL Documentation**: https://python.langchain.com/docs/expression_language/
- **Structured Output Guide**: https://python.langchain.com/docs/modules/model_io/chat/structured_output/
- **Async Programming**: https://python.langchain.com/docs/guides/async
- **Error Handling**: https://docs.python.org/3/library/asyncio-exceptions.html

---

## 🧪 Testing the Improvements

Run the updated examples:
```bash
# See structured output in action
uv run async_examples/async_planner_agent.py

# See LCEL composition
uv run async_examples/async_research_agent.py

# See error handling with mixed valid/invalid inputs
# (Look for the error handling demo in the output)
```

---

## ✅ Summary

The async examples now follow **LangChain 1.0 best practices**:

1. **Structured Output** - Type-safe Pydantic models
2. **LCEL Composition** - Declarative chain composition with `|`
3. **Error Handling** - Comprehensive try/except with logging
4. **Type Hints** - Full type annotations
5. **Logging** - Structured logging throughout
6. **Metadata** - Track duration, timestamps, errors

**Result:** More robust, maintainable, and production-ready code! 🚀
