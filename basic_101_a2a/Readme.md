# basic_101_a2a: Simple and Efficient Multi-Agent Communication  

![Demo Screenshot](https://github.com/carloderossi/a2a/blob/main/basic_101_a2a/Screenshot%202025-10-01%20142544.jpg)

## 🚀 Overview  

Seamless Multi-Agent Communication Made Easy.  
**basic_101_a2a** enables efficient interaction between Research and Planner agents using the **A2A protocol specification**, streamlining multi-agent workflow orchestration.  

Whether you want to process research queries, generate structured plans, or chain multiple agents together in complex workflows, this project provides a lightweight foundation to accelerate development.  

---

## ✨ Key Features & Benefits  

- **🔶 Multi-Agent Communication**  
  Seamless A2A-compliant interaction between Research and Planner agents, enabling interoperability and chaining across workflows.  

- **✨ Research-Oriented Queries**  
  Powered by **Ollama LLM**, process research queries and generate structured, actionable plans.  

- **🗂️ Service Discovery & Chaining**  
  Simplify multi-agent orchestration with **Agent Cards** for agent availability verification and chaining/pipelining of tasks.  

- **💡 Actionable Plan Generation**  
  Transform raw research results into executable, structured plans, with HTTP-based agent-to-agent communication.  

- **🔄 Interoperability**  
  Designed to integrate seamlessly with diverse software components, minimizing the typical pain points of multi-agent communication.  

---

## 🛠️ Addressing Developer Pain Points  

- **Interoperability Challenges**  
  Multi-agent systems often struggle with component integration—this tool ensures **A2A protocol compliance** for smoother interoperability.  

- **Protocol Complexity**  
  Designing robust communication protocols is difficult—pre-built executable classes (Research & Planner agents) save time and reduce errors.  

- **Complex Workflow Management**  
  Service discovery and chaining mechanisms simplify orchestration and debugging of multi-agent workflows.  

---

## 📊 Technical Details  

| Component      | Details |
|----------------|---------|
| ⚙️ **Architecture** | Python-based, leverage [Google Agent-to-Agent (A2A) protocol](https://a2a-protocol.org/latest/guides/) |
| 📄 **Documentation** | See comments for functions and classes. |
| 📦 **Dependencies** | See [requirements.txt](https://github.com/carloderossi/a2a/blob/main/requirements.txt). |

---

## 📸 Example Run  

Here’s an example of the project in action:

![Example Running](https://github.com/carloderossi/a2a/blob/main/basic_101_a2a/Screenshot%202025-10-01%20142544.jpg)

---

## ⚡ Getting Started  

### Prerequisites  
- Python 3.11+  
- [Ollama LLM](https://ollama.com/) installed and running  
- Currently using llama3.1

### Run the example
- ensure ollma running and serving confgured SLM (like llma3.1)
- research_agent.cmd
- planner_agent.cmd
- client_agent.cmd

### Clone Repository  
```bash
git clone https://github.com/carloderossi/a2a.git
cd a2a/basic_101_a2a
