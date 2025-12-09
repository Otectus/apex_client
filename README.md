# MemoryPlus for ApexGPT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Supercharge your local AI with temporal memory! **MemoryPlus** is a plugin for `ApexGPT` (a PyGPT client) that uses a temporal knowledge graph to remember conversations, understand context evolution, and provide deeply insightful responses.

> *A quick look at MemoryPlus in action within ApexGPT.*
> 
<img width="1863" height="638" alt="Screenshot_20251209_013749" src="https://github.com/user-attachments/assets/edbc0a90-927d-4abe-9f85-eaf9bb731d53" />
<img width="1667" height="799" alt="Screenshot_20251209_013703" src="https://github.com/user-attachments/assets/beaaa6f3-79d0-454c-bd0a-a2453db6382e" />
<img width="1667" height="799" alt="Screenshot_20251209_013723" src="https://github.com/user-attachments/assets/e910873b-9a07-4277-a6bc-d96e983205fe" />


## ✨ What is This?

Unlike traditional vector databases that only store static facts, MemoryPlus, powered by **Graphiti**, captures *when* things happened. This allows your AI to:
-   Recall the history of a topic.
-   Understand how your relationship with it evolves.
-   Inject relevant, time-aware context into every conversation.

## ✅ Key Features

-   **Two Database Backends**: Choose between **Neo4j** (robust, for scale) or **Kuzu** (zero-config, file-based).
-   **Six Powerful Memory Modes**: Analyze conversations through different lenses like `Productivity`, `Research`, or `Identity`.
-   **Intelligent Auto-Processing**: Automatically sanitizes inputs, tags topics and emotions, and prunes trivial memories.
-   **Manual Control**: Use simple commands like `/remember_this` and `/forget_that` to guide the AI's memory.
-   **RAG Prioritization**: Option to make the knowledge graph the primary source of truth over the default vector store.

## 🚀 Getting Started

### Prerequisites
*   **Python 3.10+**
*   **Git**
*   **Docker** (Optional, only if you want to use the Neo4j backend)

### 1. Clone the Repository
First, clone the `apex_client` repository to your local machine.

```bash
git clone https://github.com/Otectus/apex_client.git
cd apex_client
```

### 2. Install Dependencies
Install the required Python packages using the included `requirements.txt` file.

```bash
# This single command installs everything needed for the plugin
pip install -r apex/plugins/MemoryPlus/requirements.txt
```

### 3. Set Up the Database (Choose One)

#### 🔹 Kuzu (Recommended for Quick Start)
-   **No setup needed!** Kuzu is a file-based database that works out of the box.

#### 🔹 Neo4j (For Advanced Users)
1.  Ensure Docker is running on your system.
2.  Create a `docker-compose.yml` file in the `apex_client` root directory with the following content:
    <details>
      <summary>Click to expand docker-compose.yml</summary>

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
    </details>

3.  Start the container from your terminal:
    ```bash
    docker-compose up -d
    ```

### 4. Run the Application

Execute the custom launcher. It is already optimized to load only the MemoryPlus plugin.

```bash
python ApexGPT.py
```

## ⚙️ Configuration
Once the application is running, configure the plugin from the UI:

1.  Go to **Settings → Plugins → MemoryPlus (Graphiti)**.
2.  **Critical Settings**:
    -   **Database Backend**: Choose `Neo4j` or `Kuzu`.
    -   **Kuzu Path**: If using Kuzu, set the storage path. A good default is `~/apex/memories/kuzu`.
        -   **Windows**: `C:\Users\YourUser\apex\memories\kuzu`
        -   **macOS/Linux**: `~/apex/memories/kuzu`
    -   **Disable Default Vector Store**: ✅ Check this to prioritize Graphiti memory over PyGPT's standard RAG.
    -   **Link DB to Preset**: ✅ Check to keep memories separate for each of your presets.

## 💬 Usage
-   **Chat normally!** Memories are ingested automatically.
-   **Use commands** like `/remember_this` and `/forget_that` for manual control.
-   **Check the logs** in the PyGPT console to see the plugin fetching and storing memories in real-time.

## ❓ Troubleshooting

| Issue | Fix |
|---|---|
| **"Connection refused" (Neo4j)** | Run `docker ps` to confirm the `graphiti-neo4j` container is running. |
| **Kuzu files not saved** | Ensure the directory you set in `kuzu_path` exists and is writable. You can create it manually. |
| **No insights generated** | Verify the "Insight Model" (e.g., `gpt-4o`) is configured with a valid API key in PyGPT's main settings. |

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or pull requests, please feel free to get involved.

1.  **Fork the repository.**
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a **Pull Request**.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
