import atexit
import json
import os
import sys
import subprocess
import threading
import queue
import time
import re
from datetime import datetime
from typing import Optional, List, Tuple

from pygpt_net.plugin.base.plugin import BasePlugin
from pygpt_net.core.events import Event
from .memory_engine.client import MemoryEngineClient, ENGINE_MODES

class Plugin(BasePlugin):
    def __init__(self, *args, **kwargs):
        super(Plugin, self).__init__(*args, **kwargs)
        self.id = "MemoryPlus"
        self.name = "MemoryPlus (Graphiti)"
        self.description = "Advanced Temporal Memory using Graphiti with Active Insight Analysis."
        self.type = ["memory"]
        self.order = 90
        self.prefix = "MemoryPlus"
        self.memory_buffer = None
        self.tabs = {}
        self.ingest_queue: Optional[queue.Queue] = None
        self.ingest_thread: Optional[threading.Thread] = None
        self.ingest_stop_event: Optional[threading.Event] = None
        self.engine_client = None
        self.engine_restart_attempted = False

    def init_options(self):
        """Initialize options and tabs"""
        # General
        self.add_option("auto_ingest", "bool", value=True,
                        label="Auto-Ingest",
                        description="Automatically save conversations to memory after each interaction.",
                        tab="general")
        self.add_option("engine_mode", "combo", value="auto",
                        label="Engine Mode",
                        description="Choose how Graphiti should be executed (persistent worker or per-call subprocess).",
                        keys=list(ENGINE_MODES),
                        tab="general")
        self.add_option("inject_context", "bool", value=True,
                        label="Inject Context",
                        description="Inject relevant memories into the system prompt before generating a response.",
                        tab="general")
        self.add_option("search_depth", "int", value=10,
                        label="Context Limit",
                        description="The maximum number of relevant memories to retrieve for context.",
                        min=1, max=100,
                        tab="general")
        self.add_option("disable_default_vectors", "bool", value=False,
                        label="Disable Default Vector Store",
                        description="If checked, an instruction will be added to prioritize Graphiti memory over other context sources.",
                        tab="general")

        # Database
        self.add_option("driver_type", "combo", value="Neo4j",
                        label="Database Backend",
                        description="Select the Graph Database backend for storing memories.",
                        keys=["Neo4j", "Kuzu"],
                        tab="database")
        self.add_option("link_to_preset", "bool", value=True,
                        label="Link DB to Preset",
                        description="If enabled, creates/selects a database named after the active PyGPT Preset, isolating memories per preset.",
                        tab="database")
        # Neo4j
        self.add_option("db_uri", "text", value="bolt://localhost:7687",
                        label="Neo4j URI",
                        description="The connection URI for the Neo4j database.",
                        tab="database",
                        advanced=False)
        self.add_option("db_user", "text", value="neo4j",
                        label="Neo4j User",
                        description="The username for Neo4j authentication.",
                        tab="database",
                        advanced=False)
        self.add_option("db_pass", "text", value="password",
                        label="Neo4j Password",
                        description="The password for Neo4j authentication.",
                        secret=True,
                        tab="database",
                        advanced=False)
        self.add_option("db_name", "text", value="neo4j",
                        label="Database Name (Fallback)",
                        description="The default Neo4j database name to use if not linking to a preset.",
                        tab="database",
                        advanced=False)
        # Kuzu
        self.add_option("kuzu_path", "text", value=os.path.join(os.environ.get("HOME", ""), ".apex", "memories"),
                        label="Kuzu Storage Path",
                        description="The root directory where Kuzu database files will be stored.",
                        tab="database",
                        advanced=True)

        # Models
        self.add_option("memory_mode", "combo",
                        value="Chatbot",
                        label="Memory Mode",
                        description="Select the active memory analysis mode. This determines the lens through which conversations are analyzed for insights.",
                        keys=["Identity", "Assistant", "Chatbot", "Productivity", "Research", "Discourse"],
                        tab="models")
        self.add_option("insight_model", "combo", value="gpt-4o",
                        label="Insight Model",
                        description="The model used for generating analytical insights from conversations.",
                        use="models",
                        tab="models")
        self.add_option("llm_model", "combo", value="gpt-4o",
                        label="Graphiti Internal Model",
                        description="The model used by the Graphiti backend for its internal graph-building operations.",
                        use="models",
                        tab="models")
        self.add_option("llm_max_tokens", "int", value=8192,
                        label="Max Context Tokens",
                        description="Maximum tokens for the internal LLM's context window.",
                        min=1024, max=128000,
                        tab="models")
        self.add_option("embedding_provider", "combo",
                        value="OpenAI",
                        label="Embedding Provider",
                        description="The service provider for generating vector embeddings.",
                        keys=["OpenAI", "Ollama", "Google"],
                        tab="models")
        self.add_option("embedding_model", "combo",
                        value="text-embedding-3-small",
                        label="Embedding Model",
                        description="The specific model used to create vector embeddings for semantic search.",
                        keys=[
                            "text-embedding-3-small", "text-embedding-3-large", "nomic-embed-text",
                            "mxbai-embed-large", "all-minilm", "models/text-embedding-004", "models/gemini-embedding-001"
                        ],
                        tab="models")
        self.add_option("override_base_url", "text", value="",
                        label="Override Base URL",
                        description="(Advanced) Override the base URL for the Graphiti Internal Model.",
                        tab="models",
                        advanced=True)
        self.add_option("override_api_key", "text", value="",
                        label="Override API Key",
                        description="(Advanced) Override the API key for the Graphiti Internal Model.",
                        secret=True,
                        tab="models",
                        advanced=True)
        
        # Sanitization
        self.add_option("sanitize_tool_calls", "bool", value=True,
                        label="Sanitize Tool Calls",
                        description="Strip tool usage syntax (e.g., <tool_code>) from memories to focus on conversational content.",
                        tab="sanitization")
        self.add_option("sanitize_code_blocks", "bool", value=True,
                        label="Sanitize Code Blocks",
                        description="Strip markdown code blocks (```...```) from memories.",
                        tab="sanitization")
        self.add_option("preserve_tagged_code", "bool", value=True,
                        label="Preserve Tagged Code",
                        description="Keep code blocks tagged with [KEEP_CODE].",
                        tab="sanitization")
        self.add_option("max_memory_length", "int", value=4096,
                        label="Max Memory Length",
                        description="Truncate memories to this token length.",
                        min=100, max=10000,
                        tab="sanitization")

        # Intelligence
        self.add_option("enable_emotion_tagging", "bool", value=True,
                        label="Enable Emotion Tagging",
                        description="Automatically detect and tag memories with emotional context (e.g., [EMOTION: amused]).",
                        tab="intelligence")
        self.add_option("emotion_sensitivity", "combo", value="Medium",
                        label="Emotion Sensitivity",
                        description="Adjust how aggressively emotions are tagged.",
                        keys=["Low", "Medium", "High"],
                        tab="intelligence")
        self.add_option("enable_topic_tagging", "bool", value=True,
                        label="Enable Topic Tagging",
                        description="Automatically tag memories with relevant topics (e.g., [TOPIC: linux]).",
                        tab="intelligence")
        self.add_option("enable_vibe_scoring", "bool", value=False,
                        label="Enable Vibe Scoring",
                        description="Enable a vibe score (e.g., 0.9) for emotional tone.",
                        tab="intelligence")

        # Lifecycle
        self.add_option("auto_prune_low_value", "bool", value=False,
                        label="Auto-Prune Low-Value Memories",
                        description="Automatically remove memories that are deemed trivial or low-value (e.g., simple greetings).",
                        tab="lifecycle")
        self.add_option("low_value_threshold", "int", value=3,
                        label="Low-Value Threshold",
                        description="Minimum number of words a memory must have to be considered worth retaining.",
                        min=1, max=50,
                        tab="lifecycle")
        self.add_option("manual_memory_flagging", "bool", value=True,
                        label="Enable Manual Memory Flagging",
                        description="Allow manual flagging of memories via commands like /remember_this and /forget_that.",
                        tab="lifecycle")
        self.add_option("memory_expiry_days", "int", value=0,
                        label="Memory Expiry (Days)",
                        description="Automatically delete memories older than this number of days. Set to 0 to disable expiry.",
                        min=0, max=3650,
                        tab="lifecycle")

        # Advanced
        self.add_option("custom_sanitization_rules", "text", value="",
                        label="Custom Sanitization Rules",
                        description="Regex patterns separated by semicolons to apply during sanitization.",
                        tab="advanced")
        self.add_option("custom_memory_tags", "text", value="",
                        label="Custom Memory Tags",
                        description="Custom tags (comma-separated) to apply to all memories.",
                        tab="advanced")
        self.add_option("insight_model_temperature", "float", value=0.3,
                        label="Insight Model Temperature",
                        description="Adjust creativity of insight generation.",
                        min=0.0, max=1.0, step=0.1,
                        tab="advanced")
        self.add_option("memory_review_interval", "int", value=7,
                        label="Memory Review Interval (Days)",
                        description="Prompt the user to review memories every X days (0 = disabled).",
                        min=0, max=365,
                        tab="advanced")
        self.add_option("enable_memory_feedback", "bool", value=True,
                        label="Enable Memory Feedback",
                        description="Let the user rate memories via 👍/👎.",
                        tab="advanced")
        self.add_option("memory_search_depth", "int", value=10,
                        label="Memory Search Depth",
                        description="Number of memories to retrieve during search.",
                        min=1, max=100,
                        tab="advanced")
        self.add_option("ingest_queue_size", "int", value=50,
                        label="Ingestion Queue Size",
                        description="Maximum number of pending ingestion items. 0 = unlimited.",
                        min=0, max=1000,
                        tab="advanced")
        self.add_option("ingest_overflow_policy", "combo", value="drop_new",
                        label="Ingestion Overflow Policy",
                        description="When the ingestion queue is full: drop new item, drop oldest item, or block until space is free.",
                        keys=["drop_new", "drop_oldest", "block"],
                        tab="advanced")
        self.add_option("ingest_batch_max_items", "int", value=5,
                        label="Ingestion Batch Size",
                        description="Maximum number of items to process together from the queue.",
                        min=1, max=100,
                        tab="advanced")
        self.add_option("ingest_batch_max_delay_ms", "int", value=250,
                        label="Ingestion Batch Delay (ms)",
                        description="Maximum time to wait for additional items before processing a batch.",
                        min=0, max=5000,
                        tab="advanced")


    def init_tabs(self) -> dict:
        """Initialize provider tabs"""
        tabs = {}
        tabs["general"] = "General"
        tabs["database"] = "Database"
        tabs["models"] = "Models"
        tabs["sanitization"] = "Sanitization"
        tabs["intelligence"] = "Intelligence"
        tabs["lifecycle"] = "Lifecycle"
        tabs["advanced"] = "Advanced"
        return tabs

    def attach(self, window):
        super(Plugin, self).attach(window)
        self.window = window
        self.tabs = self.init_tabs()
        self.init_options()
        self._init_engine()
        self._start_ingest_worker()
        self.log("Plugin attached. Ready for operations.")

    def detach(self, *args, **kwargs):
        self._stop_ingest_worker()
        self._shutdown_engine()
        super(Plugin, self).detach(*args, **kwargs)

    def handle(self, event: Event, *args, **kwargs):
        if event.name == Event.MODELS_CHANGED:
            self.refresh_option("llm_model")
            self.refresh_option("insight_model")
        elif event.name == Event.CTX_BEFORE:
            self._on_ctx_before(event)
        elif event.name == Event.SYSTEM_PROMPT:
            self._on_system_prompt(event)
        elif event.name == Event.CTX_AFTER:
            self._on_ctx_after(event)

    def _get_model_config(self, option_name="llm_model"):
        model_id = self.get_option_value(option_name)
        config = {"model": model_id, "api_key": "", "base_url": "", "provider": "openai"}

        if self.window and self.window.core.models.has(model_id):
            model_item = self.window.core.models.get(model_id)
            client_args = self.window.core.models.prepare_client_args(model=model_item)
            config["api_key"] = client_args.get("api_key", "")
            config["base_url"] = client_args.get("base_url", "")
            config["provider"] = model_item.provider

        if option_name == "llm_model":
            if self.get_option_value("override_api_key"):
                config["api_key"] = self.get_option_value("override_api_key")
            if self.get_option_value("override_base_url"):
                config["base_url"] = self.get_option_value("override_base_url")
        return config

    def _get_group_id(self):
        if self.get_option_value("link_to_preset"):
            preset_id = self.window.core.config.get('preset')
            if preset_id:
                return preset_id
        return self.get_option_value("db_name")

    def _build_engine_config(self):
        main_model_config = self._get_model_config("llm_model")
        insight_model_config = self._get_model_config("insight_model")
        google_key = self.window.core.config.get("api_key_google") or ""

        return {
            # DB
            "driver_type": self.get_option_value("driver_type"),
            "uri": self.get_option_value("db_uri"),
            "user": self.get_option_value("db_user"),
            "password": self.get_option_value("db_pass"),
            "kuzu_path": self.get_option_value("kuzu_path"),
            # LLM
            "llm": {
                "provider": main_model_config["provider"],
                "model": main_model_config["model"],
                "base_url": main_model_config["base_url"],
                "api_key": main_model_config["api_key"],
                "max_tokens": self.get_option_value("llm_max_tokens"),
            },
            # Insight LLM
            "insight_llm": {
                "provider": insight_model_config["provider"],
                "model": insight_model_config["model"],
                "base_url": insight_model_config["base_url"],
                "api_key": insight_model_config["api_key"],
            },
            # Embedding
            "embedding": {
                "provider": self.get_option_value("embedding_provider"),
                "model": self.get_option_value("embedding_model"),
                "google_api_key": google_key
            },
            # New config options
            "sanitization": {
                "sanitize_tool_calls": self.get_option_value("sanitize_tool_calls"),
                "sanitize_code_blocks": self.get_option_value("sanitize_code_blocks"),
                "preserve_tagged_code": self.get_option_value("preserve_tagged_code"),
                "max_memory_length": self.get_option_value("max_memory_length"),
                "custom_sanitization_rules": self.get_option_value("custom_sanitization_rules"),
            },
            "intelligence": {
                "enable_emotion_tagging": self.get_option_value("enable_emotion_tagging"),
                "emotion_sensitivity": self.get_option_value("emotion_sensitivity"),
                "enable_topic_tagging": self.get_option_value("enable_topic_tagging"),
                "enable_vibe_scoring": self.get_option_value("enable_vibe_scoring"),
            },
            "lifecycle": {
                "auto_prune_low_value": self.get_option_value("auto_prune_low_value"),
                "low_value_threshold": self.get_option_value("low_value_threshold"),
                "manual_memory_flagging": self.get_option_value("manual_memory_flagging"),
                "memory_expiry_days": self.get_option_value("memory_expiry_days"),
            },
            "advanced": {
                "custom_memory_tags": self.get_option_value("custom_memory_tags"),
                "insight_model_temperature": self.get_option_value("insight_model_temperature"),
                "memory_review_interval": self.get_option_value("memory_review_interval"),
                "enable_memory_feedback": self.get_option_value("enable_memory_feedback"),
                "memory_search_depth": self.get_option_value("memory_search_depth"),
            }
        }

    def _get_runner_cmd(self, operation: str, **kwargs):
        runner_path = os.path.join(os.path.dirname(__file__), "runner.py")
        config = self._build_engine_config()

        kwargs["group_id"] = self._get_group_id()

        cmd = [sys.executable, runner_path, "--config", json.dumps(config), "--operation", operation]
        for k, v in kwargs.items():
            cmd.append(f"--{k}")
            cmd.append(str(v))
        return cmd

    def _run_subprocess(self, cmd, background=False):
        self.log(f"Executing: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            response = None
            if result.stdout.strip():
                try:
                    response = json.loads(result.stdout)
                except json.JSONDecodeError: pass

            if result.returncode != 0:
                err_msg = result.stderr.strip() or (response and response.get("error")) or "Unknown runner error"
                msg = f"Runner failed: {err_msg}"
                self.log(f"[ERROR] {msg}") if background else self.error(msg)
                return None
            return response
        except Exception as e:
            msg = f"Subprocess error: {e}"
            self.log(f"[ERROR] {msg}") if background else self.error(msg)
            return None

    def _init_engine(self):
        if self.get_option_value("engine_mode") not in ["persistent", "auto"]:
            return
        self.engine_restart_attempted = False
        self._ensure_engine_client()
        client = self.engine_client
        if client and not client.start():
            self.error("Failed to start persistent Graphiti worker. Falling back to subprocess mode.")

    def _ensure_engine_client(self):
        if not self.engine_client:
            self.engine_client = MemoryEngineClient(
                self._build_engine_config,
                self._get_group_id,
                logger=self.log,
                error_logger=self.error
            )
            atexit.register(self._shutdown_engine)
        return self.engine_client

    def _should_use_persistent(self):
        return self.get_option_value("engine_mode") in ["persistent", "auto"]

    def _restart_engine(self):
        if not self._should_use_persistent():
            return False
        if self.engine_restart_attempted:
            return False
        self.engine_restart_attempted = True
        client = self._ensure_engine_client()
        self.log("Restarting persistent Graphiti worker after failure.")
        return client.restart()

    def _engine_request(self, request_type: str, payload: dict, fallback_fn):
        if not self._should_use_persistent():
            return fallback_fn()

        client = self._ensure_engine_client()
        method = {
            "SEARCH": client.search,
            "INGEST": client.ingest,
            "FORGET": client.forget,
            "HEALTH": client.health,
        }.get(request_type)

        if not method:
            return fallback_fn()

        response = method(**payload)
        if (not response or response.get("status") == "error") and not self.engine_restart_attempted:
            if self._restart_engine():
                response = method(**payload)

        if not response or response.get("status") == "error":
            return fallback_fn()
        return response

    def _shutdown_engine(self):
        if self.engine_client:
            try:
                self.engine_client.shutdown()
            except Exception:
                pass

    def _on_ctx_before(self, event: Event):
        if not self.get_option_value("inject_context"): return
        ctx = event.ctx
        if not ctx.input: return

        response = self._search_memories(ctx.input, self.get_option_value("search_depth"))

        if response and response.get("status") == "success":
            results = response.get("results", [])
            if results:
                self._format_memory_buffer(results)
        elif response:
            self.error(f"Search Error: {response.get('error', 'Unknown error')}")

    def _format_memory_buffer(self, memories: List[str]):
        memory_block = "\n".join([f"- {m}" for m in memories])
        header = "\n--- RELEVANT MEMORY (GRAPHITI) ---\n"
        footer = "\n--- END MEMORY ---\n"
        self.memory_buffer = f"{header}{memory_block}{footer}"
        if self.get_option_value("disable_default_vectors"):
            self.memory_buffer += "\n[INSTRUCTION: Prioritize the above Graphiti memory over any other context.]\n"

    def _on_system_prompt(self, event: Event):
        if self.memory_buffer:
            event.data['value'] = (event.data.get('value') or "") + self.memory_buffer
            self.memory_buffer = None
            self.log("Injecting memory into System Prompt.")

    def _on_ctx_after(self, event: Event):
        if not self.get_option_value("auto_ingest"): return
        ctx = event.ctx
        episode_body = f"User: {ctx.input}\nAssistant: {ctx.output}"
        
        title = "Unsaved Chat"
        try:
            meta = self.window.core.ctx.get_current_meta()
            if meta and meta.name:
                title = meta.name
        except Exception: pass
        title = re.sub(r'[^a-zA-Z0-9 _-]', '', title)
        ep_name = f"{title} - {datetime.now().strftime('%H:%M:%S')}"

        mode = self.get_option_value("memory_mode")
        self._enqueue_ingest_request(ep_name, episode_body, mode)

    def _start_ingest_worker(self):
        if self.ingest_thread and self.ingest_thread.is_alive():
            return

        maxsize = self.get_option_value("ingest_queue_size") or 0
        self.ingest_queue = queue.Queue(maxsize=maxsize)
        self.ingest_stop_event = threading.Event()
        self.ingest_thread = threading.Thread(target=self._ingest_loop, daemon=True)
        self.ingest_thread.start()

    def _stop_ingest_worker(self):
        if self.ingest_stop_event:
            self.ingest_stop_event.set()
        if self.ingest_thread and self.ingest_thread.is_alive():
            self.ingest_thread.join(timeout=2)
        if self.ingest_queue:
            try:
                while True:
                    dropped = self.ingest_queue.get_nowait()
                    self.log(f"[WARN] Ingest worker stopped. Dropping pending item: {dropped[0]}")
                    self.ingest_queue.task_done()
            except queue.Empty:
                pass
        self.ingest_thread = None
        self.ingest_queue = None
        self.ingest_stop_event = None

    def _enqueue_ingest_request(self, name: str, content: str, mode: str):
        if not self.ingest_queue:
            self._start_ingest_worker()

        overflow_policy = self.get_option_value("ingest_overflow_policy")
        item = (name, content, mode)

        if overflow_policy == "block":
            while self.ingest_stop_event and not self.ingest_stop_event.is_set():
                try:
                    self.ingest_queue.put(item, timeout=0.5)
                    return
                except queue.Full:
                    continue
            self.log(f"[WARN] Ingest worker stopping. Dropping item: {name}")
            return

        try:
            self.ingest_queue.put(item, block=False)
        except queue.Full:
            if overflow_policy == "drop_oldest":
                try:
                    dropped = self.ingest_queue.get_nowait()
                    self.ingest_queue.task_done()
                    self.log(f"[WARN] Ingest queue full. Dropping oldest item: {dropped[0]}")
                except queue.Empty:
                    pass
                try:
                    self.ingest_queue.put_nowait(item)
                except queue.Full:
                    self.log(f"[WARN] Ingest queue full. Dropping new item: {name}")
            else:
                self.log(f"[WARN] Ingest queue full. Dropping new item: {name}")

    def _ingest_loop(self):
        while self.ingest_stop_event and not self.ingest_stop_event.is_set():
            try:
                first_item: Tuple[str, str, str] = self.ingest_queue.get(timeout=0.5)  # type: ignore
            except queue.Empty:
                continue

            batch = [first_item]
            max_items = max(1, int(self.get_option_value("ingest_batch_max_items") or 1))
            max_delay = max(0, int(self.get_option_value("ingest_batch_max_delay_ms") or 0)) / 1000
            start = time.monotonic()

            while len(batch) < max_items:
                remaining = max_delay - (time.monotonic() - start)
                if remaining <= 0:
                    break
                try:
                    next_item = self.ingest_queue.get(timeout=remaining)
                    batch.append(next_item)
                except queue.Empty:
                    break

            for name, content, mode in batch:
                self._process_ingest(name, content, mode)
                self.ingest_queue.task_done()

    def _process_ingest(self, name: str, content: str, mode: str):
        response = self._engine_request(
            "INGEST",
            {"name": name, "content": content, "mode": mode},
            lambda: self._run_subprocess(self._get_runner_cmd("add", name=name, content=content, mode=mode), background=True)
        )
        if response and response.get("status") == "success":
            self.log(f"Ingested: {name} [Mode: {mode}]")
        elif response:
            self.log(f"[ERROR] Ingest Error: {response.get('error')}")

    def _search_memories(self, query: str, limit: int):
        return self._engine_request(
            "SEARCH",
            {"query": query, "limit": limit},
            lambda: self._run_subprocess(self._get_runner_cmd("search", query=query, limit=limit))
        )
