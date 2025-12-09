# MemoryPlus (Graphiti)

A temporal knowledge graph plugin for Py-GPT that captures *when* facts occurred and how relationships evolve over time. Unlike static vector storage, MemoryPlus enables context-aware responses that understand the *history* and *evolution* of conversations.

## ✨ Features
- **Two Database Backends**: Neo4j (for production) or **Kuzu** (file-based, zero-config) - **no Docker needed!**
- **Memory Modes**: 6 analysis styles (`Identity`, `Assistant`, `Chatbot`, `Productivity`, `Research`, `Discourse`)
- **Auto-Processing**: Sanitizes code/tool calls, tags emotions/topics, and prunes low-value memories
- **Manual Overrides**: `/remember_this` and `/forget_that` commands for fine-grained control
- **Prioritization**: Replace Py-GPT's default RAG system with Graphiti memory

## 🚀 Installation
### 1. Python Dependencies
```bash
pip install -r requirements.txt
# Or manually:
pip install graphiti-core[neo4j] nest_asyncio
```

### 2. Database Setup
#### 🔹 **Neo4j** (Docker recommended):
Create `docker-compose.yml`:
```yaml
services:
  neo4j:
    image: neo4j:5.26.0
    container_name: graphiti-neo4j
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=apoc,graph-data-science
      - NEO4J_dbms_security_procedures_unrestricted=gds.*,apoc.*
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - ./neo4j_data:/data
```

Start with:
```bash
docker-compose up -d
```

#### 🔹 **Kuzu** (Zero-Config Local Option):
- Set `kuzu_path` in plugin settings (e.g., `~/.apex/memories/kuzu`)
- Kuzu automatically creates DB files there—**no Docker, no server, no setup!**

> 💡 **When to choose which?**
>- Use **Neo4j** for large-scale deployments or team environments.
>- Use **Kuzu** for local/personal use (single-file DB, no configuration).

## ⚙️ Configuration in Py-GPT
1. Go to **Settings → Plugins → MemoryPlus (Graphiti)**

2. **Critical Settings**:
   - **Database Backend**: Choose `Neo4j` or `Kuzu`
     - For Neo4j: Set URI (`bolt://localhost:7687`), user (`neo4j`), password (`password`)
     - For Kuzu: Set `kuzu_path` to a local writable directory
   - **Disable Default Vector Store**: ✅ Check this to **prioritize Graphiti memory over RAG**
   - **Memory Modes**: Select analysis style (e.g., `Productivity` generates task dependencies)
   - **Link DB to Preset**: ✅ Check to isolate memories per Py-GPT configuration preset

## 💬 Usage
- **Auto-Remembering**: Every message is saved automatically to the graph without manual saves.
- **Context Injection**: Memories automatically surface during responses (e.g., "You discussed Linux networking last week—did you test those commands?")
- **Manual Overrides**:
  - `/remember_this` → Force-retain this memory (bypasses pruning)
  - `/forget_that` → Flag for immediate pruning
- **Debugging**: Check Py-GPT console logs for messages like `Ingested: [name]` or `Retrieved X memories for query`

## 🤖 How It Works (Technical Deep Dive)
1. **Sanitization**:
   - Strips tool calls (`<tool_code>...`), code blocks (unless `[KEEP_CODE]` tagged), and truncates content
2. **Intelligence Layer**:
   - Adds `[EMOTION: excited]` or `[TOPIC: Python]` tags
   - For `Productivity` mode: Generates task dependency maps
3. **Storage**:
   - Neo4j: Creates databases per preset if "Link DB to Preset" is enabled
   - Kuzu: Stores in local file directory (e.g., `~/.apex/memories/kuzu`)
4. **Retrieval**:
   - Searches by semantic relevance + time decay (newer memories prioritized)
   - Injects results into system prompt before generating responses

## ❓ Troubleshooting
| Issue | Fix |
|-------|-----|
| "Connection refused" (Neo4j) | Run `docker ps` to confirm container is up. Check credentials in Py-GPT settings. |
| Kuzu files not saved | Ensure `kuzu_path` exists and is writable (`mkdir -p ~/.apex/memories/kuzu`). |
| No insights generated | Verify the "Insight Model" (e.g., `gpt-4o`) has API keys configured in Py-GPT. |
| "ImportError: kuzu" | Kuzu is built into Graphiti—no extra install needed. Reinstall with `pip install -U graphiti-core` |

## 📚 Further Reading
- [Graphiti Documentation](https://graphiti.ai)
- [Neo4j Developer Hub](https://neo4j.com/developer)
- [Py-GPT Plugins Guide](https://py-gpt-docs.readthedocs.io)
