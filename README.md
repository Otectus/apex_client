[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![version](https://img.shields.io/badge/version-1.0.0-green)](./)

# MemoryPlus: Temporal Memory for ApexGPT 🧠✨

**MemoryPlus is more than a database; it's a temporal reasoning engine for your AI.** It empowers [ApexGPT](https://github.com/Otectus/apex_client) to remember *when* things happened, turning flat conversations into a dynamic, time-aware knowledge graph.

Powered by **Graphiti**, this plugin enables your AI to understand context, recall historical details, and build evolving relationships.

> *"MemoryPlus doesn’t just store facts; it remembers the story of your interactions."*

---

## 🌟 Why MemoryPlus?

While traditional vector search finds similar facts, MemoryPlus adds the crucial dimension of **time**. This unlocks deeper reasoning capabilities for your AI, allowing it to:

-   **Recall Conversation History:** *"What was my original idea for this project last month?"*
-   **Track Evolving Narratives:** *"How has the user's opinion on Topic X changed over our last three conversations?"*
-   **Inject Relevant, Timely Context:** *"You mentioned you were busy last week; is now a better time to discuss this?"*

### Real-World Use Cases

| Scenario | How MemoryPlus Helps |
| :--- | :--- |
| **Personalized AI Companion** | Remembers your goals, preferences, and key life events. |
| **Long-Term Project Assistant** | Tracks decisions, follows up on action items, and recalls project history. |
| **Research & Analysis Tool** | Connects sources, facts, and contradictions across time. |
| **Creative Writing Partner** | Maintains plot continuity and character development across sessions. |
| **Customer Support Agent** | Provides personalized support by recalling a customer's full interaction history. |

---

## ⚙️ How It Works

MemoryPlus operates in a seamless, four-stage loop that stays invisible to the user:

1.  **📥 Capture:** Every exchange (or manual `/remember_this` command) is captured. Tool calls and code blocks are automatically sanitized unless tagged with `[KEEP_CODE]`.
2.  **🧠 Analyze:** The text is processed through active **Memory Modes** to extract topics, entities, and relationships.
3.  **📈 Store:** This structured data is added to the **Graphiti** knowledge graph. An intelligent **Ingestion Queue** handles high-volume chats without blocking the UI.
4.  **✨ Inject:** Before the AI replies, MemoryPlus queries the graph. A fuzzy-matching **Search Cache** ensures instant retrieval for recurring topics.

---

## ⚡ Quick Start (TL;DR)

1.  **Clone ApexGPT** (if you haven’t already):
    ```bash
    git clone https://github.com/Otectus/apex_client.git && cd apex_client
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r apex/plugins/MemoryPlus/requirements.txt
    ```
3.  **Run the App**: (Kuzu is the default zero-config backend)
    ```bash
    python ApexGPT.py
    ```
4.  **Configure in ApexGPT**:
    -   Go to **Settings → Plugins → MemoryPlus (Graphiti)**.
    -   Set your desired `Database Backend` (`Kuzu` or `Neo4j`).
    -   (Recommended) Check `Disable Default Vector Store` to make MemoryPlus the primary context source.
5.  **Start Chatting!** Memories are captured automatically.

---

## 🛠️ Full Setup Guide

### Prerequisites

-   Python 3.10+
-   Git
-   Docker (Optional, only for the Neo4j backend)

### Database Backend (Choose One)

#### 🔵 Option 1: Kuzu (Recommended for Simplicity)

-   **Zero setup required!** Kuzu is an embedded, file-based graph database.
-   By default, it stores data in `~/apex/memories/kuzu`. You can change this path in the plugin settings.

#### 🔵 Option 2: Neo4j (Recommended for Scale)

1.  Create a `docker-compose.yml` file in the `apex_client` root directory:
    ```yaml
    services:
      neo4j:
        image: neo4j:5.26.0
        container_name: graphiti-neo4j
        environment:
          - NEO4J_AUTH=neo4j/password
          - NEO4J_PLUGINS=["apoc", "graph-data-science"]
          - NEO4J_dbms_security_procedures_unrestricted=gds.*,apoc.*
        ports:
          - "7474:7474"  # Web UI
          - "7687:7687"  # Bolt Connector
        volumes:
          - ./neo4j_data:/data
    ```
2.  Start the container: `docker-compose up -d`
3.  Connect to the Neo4j browser at `http://localhost:7474` (user: `neo4j`, pass: `password`).

---

## 🧠 Memory Modes

MemoryPlus uses "modes" to change its analysis lens. Combine them for incredibly rich insights.

| Mode | Focus |
| :--- | :--- |
| **Identity** | User persona, preferences, relationships. |
| **Assistant** | AI instructions, workflows, and self-correction. |
| **Chatbot** | Balanced capture of conversational flow for general chats. |
| **Productivity** | Tasks, blockers, dependencies, and follow-ups. |
| **Research** | Facts, claims, sources, and contradictions. |
| **Discourse** | Argument structure, rhetoric, and reasoning chains. |
| **ResolveEntities** | Suggests merges for similar topics to keep the graph tidy. |
| **CustomPrompt** | Run your own analysis instructions (set in Advanced settings). |

---

## 🎛️ Configuration

### Core Settings
-   **`Database Backend`**: Switch between `Kuzu` and `Neo4j`.
-   **`Link DB to Preset`**: Keep memories separate for each of your AI presets.
-   **`Engine Mode`**: `Auto` (persistent background worker, recommended) or `Subprocess` (fire-and-forget).

### 🚀 Power User Settings (Advanced)

<details>
<summary><strong>Click to Expand Advanced Configuration</strong></summary>

#### Performance Tuning
-   **Search Cache Similarity:** (Default: `0.85`) fuzzy matching threshold. Lower it to catch more variations of a query, raise it for precision.
-   **Ingestion Queue:**
    -   **Size:** Buffer size for incoming messages.
    -   **Overflow Policy:** Choose `drop_new` (safe), `drop_oldest` (keep current context), or `block` (guarantee capture, may slow UI).

#### Sanitization & Coding
-   **Sanitize Tool Calls:** Strips `<tool_code>` to keep memory clean.
-   **Preserve Tagged Code:** If you want the AI to remember a specific snippet, have it tag the block with `[KEEP_CODE]`. Standard code blocks are otherwise summarized to save space.

#### Lifecycle Management
-   **Auto-Prune Low Value:** Automatically discards "trivial" memories (like "Hi", "Thanks") based on word count (`Low Value Threshold`).
-   **Memory Expiry:** Set a day limit to auto-delete old memories, keeping your graph relevant.

#### Reliability
-   **Embedding Fallback:** If you select `Ollama` but the model is unreachable, MemoryPlus automatically falls back to `OpenAI` to prevent errors.
</details>

---

## 💬 Usage

### Automatic & Manual Control

-   **Automatic:** Just chat normally! Memories are captured and injected into the context automatically.
-   **Manual:** Use simple commands for fine-grained control.

| Command | Description |
| :--- | :--- |
| `/remember_this` | Force the current conversation turn into long-term memory. |
| `/forget_that` | Delete the most recently stored memory. |
| `/memory_stats` | Display statistics about the current knowledge graph. |

### Visualizing the Graph

MemoryPlus builds a rich web of connections over time.

```
// A simplified view of the temporal graph:

(User: Ace)-[:SAID {ts: '2025-12-14'}]->(Message: "Let's discuss project Nova")
  |
  `--[:HAS_INTEREST]->(Topic: Nova)-[:RELATED_TO]->(Concept: AI)

(AI)-[:SAID {ts: '2025-12-15'}]->(Message: "Nova uses a temporal graph")
  |
  `--[:EXPLAINED]->(Topic: Temporal Graph)
```

---

## ❓ Troubleshooting & FAQ

| Issue | Fix |
| :--- | :--- |
| **"Connection refused" (Neo4j)** | Run `docker ps` to confirm the `graphiti-neo4j` container is running. |
| **Kuzu files not saving** | Ensure the directory you set in `kuzu_path` exists and is writable. |
| **No memories injected** | Confirm **`Inject Context`** is enabled and try clearing the search cache. |
| **Ollama Failing?** | Ensure your local Ollama instance is running. The plugin will auto-fallback to OpenAI if it fails. |

**Q: Which API keys does this use?**
A: It reuses PyGPT’s main configured credentials by default. You can override this in the plugin's "Models" tab to use a dedicated endpoint like Ollama.

**Q: Does the search cache serve stale data?**
A: No. The cache is immediately cleared after any new memory is added, ensuring fresh data is always available for the next query.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE.md` for more information.
