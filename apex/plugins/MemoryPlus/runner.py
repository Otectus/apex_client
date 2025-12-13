import asyncio
import json
import sys
import os
import argparse
import re
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Optional, Dict, Any, List, Callable

# --- IMPORTS ---
try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.llm_client import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    import openai
    import neo4j
    
    # Optional: Kuzu
    try:
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        import kuzu
    except ImportError:
        KuzuDriver = None

    # Optional: Gemini
    try:
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
    except ImportError:
        GeminiEmbedder = None

except ImportError as e:
    print(json.dumps({"error": f"ImportError: {e}. Please ensure graphiti-core is installed."}))
    sys.exit(1)

# --- CONSTANTS ---
ANALYSIS_PROMPTS = {
    "Identity": (
        "Analyze the following interaction for shifts in the AI's persona, new relationship dynamics, "
        "or reinforced traits. Focus on 'Persona Calibration'. Output a concise 'Calibration Log' summarizing these changes."
    ),
    "Assistant": (
        "Analyze the user's request for workflow patterns, repetitive tasks, or implied needs. "
        "Focus on 'Anticipatory Needs'. Output a concise 'User Profile Update'."
    ),
    "Chatbot": (
        "Analyze the interaction for both user preferences and model identity traits. "
        "Identify which context (Identity vs User) is dominant. Output a 'Relational Context Summary'."
    ),
    "Productivity": (
        "Analyze the text to extract actionable tasks, project dependencies, and potential bottlenecks. "
        "Focus on 'Workflow Optimization'. Output a concise 'Dependency Map'."
    ),
    "Research": (
        "Extract key claims, source reliability (if mentioned), and potential contradictions with established facts. "
        "Focus on 'Source Integrity'. Output a 'Semantic Validation Log'."
    ),
    "Discourse": (
        "Identify the logical structure of the argument, note any fallacies (strawman, ad hominem, etc.), "
        "and map the rhetorical strategy. Output a 'Logic & Rhetoric Analysis'."
    )
}

# --- PROCESSING LAYERS ---

def sanitize_memory(raw_text: str, config: Dict[str, Any]) -> str:
    """
    Strips tool usage and code blocks based on plugin settings.
    """
    if not isinstance(raw_text, str):
        return ""

    text = raw_text
    sanitization_config = config.get("sanitization", {})

    if sanitization_config.get("sanitize_tool_calls", True):
        text = re.sub(r'<tool_code>.*?</tool_code>', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)

    if sanitization_config.get("sanitize_code_blocks", True):
        preserve_tagged_code = sanitization_config.get("preserve_tagged_code", True)

        def _code_replacer(match: re.Match) -> str:
            leading_newline, prefix, has_keep_tag, block = match.group(1), match.group(2), match.group(3), match.group(4)
            if preserve_tagged_code and has_keep_tag:
                return f"{leading_newline}{prefix}{block}"  # Strip the [KEEP_CODE] tag but keep the code block
            return leading_newline  # Keep surrounding whitespace to avoid collapsing content

        code_pattern = re.compile(r'(^|\n)(\s*)(\[KEEP_CODE\]\s*)?(```.*?```)', re.DOTALL)
        text = re.sub(code_pattern, _code_replacer, text)

    normalize_whitespace = sanitization_config.get("normalize_whitespace", False)
    if normalize_whitespace:
        text = re.sub(r"\n{3,}", "\n\n", text)

    # Truncate to max length
    max_len = sanitization_config.get("max_memory_length", 4096)
    if len(text) > max_len:
        text = text[:max_len]

    # Custom rules
    custom_rules = sanitization_config.get("custom_sanitization_rules", "")
    if custom_rules:
        for rule in custom_rules.split(";"):
            try:
                text = re.sub(rule, "", text)
            except Exception:
                pass  # Silently ignore malformed regex

    return text.strip()


def _load_json_list(path: str) -> List[Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return []


def _save_json_list(path: str, payload: List[Any]) -> None:
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _within_window(timestamp: datetime, now: datetime, minutes: int) -> bool:
    return (now - timestamp) <= timedelta(minutes=minutes)


def evaluate_ingestion_controls(
    content: str,
    config: Dict[str, Any],
    now: Optional[datetime] = None,
    loader: Callable[[str], List[Any]] = _load_json_list,
    saver: Callable[[str, List[Any]], None] = _save_json_list,
) -> Optional[str]:
    """
    Checks ingestion guardrails. Returns a human-readable reason when ingestion
    should be skipped; otherwise returns None.
    """

    settings = config.get("ingestion_controls", {})
    now = now or datetime.utcnow()
    pending_saves: List[tuple[str, List[Any]]] = []

    # Minimum size checks
    min_chars = settings.get("min_chars")
    if isinstance(min_chars, int) and len(content.strip()) < min_chars:
        return "Content shorter than configured min_chars"

    min_words = settings.get("min_words")
    if isinstance(min_words, int) and len(re.findall(r"\S+", content)) < min_words:
        return "Content shorter than configured min_words"

    max_tokens = settings.get("max_tokens")
    if isinstance(max_tokens, int) and len(re.findall(r"\S+", content)) > max_tokens:
        return "Content exceeds configured max_tokens"

    # Rate limiting
    max_ingestions = settings.get("max_ingestions_per_minute")
    if isinstance(max_ingestions, int) and max_ingestions > 0:
        cache_path = settings.get("rate_limit_cache", "/tmp/memoryplus_ingest_rate.json")
        timestamps = []
        for ts in loader(cache_path):
            try:
                parsed = datetime.fromisoformat(ts)
                if _within_window(parsed, now, 1):
                    timestamps.append(parsed)
            except Exception:
                continue
        if len(timestamps) >= max_ingestions:
            return "Ingestion rate limit exceeded"
        timestamps.append(now)
        pending_saves.append((cache_path, [t.isoformat() for t in timestamps]))

    # Deduplication window
    dedup_minutes = settings.get("deduplication_window_minutes")
    if isinstance(dedup_minutes, int) and dedup_minutes > 0:
        cache_path = settings.get("dedup_cache", "/tmp/memoryplus_dedup.json")
        digest = sha256(content.strip().encode("utf-8")).hexdigest()
        entries = []
        seen_recently = False
        for entry in loader(cache_path):
            try:
                entry_time = datetime.fromisoformat(entry.get("timestamp", ""))
                if _within_window(entry_time, now, dedup_minutes):
                    entries.append(entry)
                    if entry.get("digest") == digest:
                        seen_recently = True
            except Exception:
                continue
        if seen_recently:
            return "Duplicate content within deduplication window"
        entries.append({"digest": digest, "timestamp": now.isoformat()})
        pending_saves.append((cache_path, entries))

    for cache_path, payload in pending_saves:
        saver(cache_path, payload)

    return None

def detect_emotion(text: str, sensitivity: str = "Medium") -> str:
    # Placeholder logic for emotion detection
    emotions = {
        "Low": {"happy": 0.2, "sad": 0.1, "angry": 0.1},
        "Medium": {"happy": 0.3, "sad": 0.2, "curious": 0.2},
        "High": {"excited": 0.4, "thoughtful": 0.3, "amused": 0.2},
    }
    # Simplified emotion tagging logic
    if sensitivity not in emotions:
        sensitivity = "Medium"
    for emotion, threshold in emotions[sensitivity].items():
        if hash(text) % 100 < threshold * 100:
            return emotion.capitalize()
    return "Neutral"

def detect_topics(text: str) -> List[str]:
    # Keywords-based topic detection
    topics = []
    keywords_to_topic = {
        "linux|bash|shell|kernel": "Linux",
        "python|flask|django|asyncio": "Python",
        "game|gaming|play|fun": "Gaming",
        "finance|money|invest|stock": "Finance",
        "anime|manga|otaku": "Anime",
        "poetry|creative|art": "Art",
        "tech|cyber|code": "Tech",
        "love|romance|feeling": "Love"
    }
    for pattern, topic in keywords_to_topic.items():
        if re.search(pattern, text, re.IGNORECASE):
            topics.append(topic)
    return topics if topics else ["Unclassified"]

def apply_intelligence_layer(content: str, original_content: str, config: Dict[str, Any]) -> str:
    """
    Applies emotion/topic tagging based on plugin settings.
    """
    intelligence_config = config.get("intelligence", {})
    
    # Emotion Tagging
    if intelligence_config.get("enable_emotion_tagging", False):
        sensitivity = intelligence_config.get("emotion_sensitivity", "Medium")
        emotion = detect_emotion(original_content, sensitivity)
        content = f"[EMOTION: {emotion}] {content}"

    # Topic Tagging
    if intelligence_config.get("enable_topic_tagging", False):
        topics = detect_topics(original_content)
        topic_tags = " ".join([f"[TOPIC: {topic}]" for topic in topics])
        content = f"{topic_tags} {content}"
        
    return content

def check_lifecycle(content: str, config: Dict[str, Any]) -> bool:
    """
    Checks if a memory should be pruned based on lifecycle settings.
    Returns True if the memory should be pruned, False otherwise.
    """
    lifecycle_config = config.get("lifecycle", {})

    # Auto-Pruning
    if lifecycle_config.get("auto_prune_low_value", False):
        threshold = lifecycle_config.get("low_value_threshold", 3)
        if len(content.strip().split()) < threshold:
            return True

    return False

# --- CONFIG HELPERS ---
def get_llm_config_params(section_conf: Dict[str, Any]) -> Dict[str, Any]:
    # (No changes needed in this function)
    provider = section_conf.get("provider", "").lower()
    model = section_conf.get("model", "gpt-4o")
    base_url = section_conf.get("base_url")
    api_key = section_conf.get("api_key")
    if provider == "ollama":
        if not base_url: base_url = "http://localhost:11434/v1"
        if not api_key: api_key = "ollama"
    if "google" in provider and not base_url:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    if not api_key and not base_url:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
    return {"api_key": api_key, "base_url": base_url, "model": model, "provider": provider}

# --- DRIVER SETUP ---
def setup_neo4j_driver(config: Dict[str, Any], group_id: str) -> None:
    # (No changes needed in this function)
    if group_id in ["neo4j", "system"]: return
    uri, user, password = config.get("uri"), config.get("user"), config.get("password")
    try:
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database="system") as session:
            session.run(f"CREATE DATABASE {group_id} IF NOT EXISTS")
        driver.close()
    except Exception: pass

def setup_kuzu_driver(config: Dict[str, Any], group_id: str) -> Optional[Any]:
    # (No changes needed in this function)
    if KuzuDriver is None:
        print(json.dumps({"error": "Kuzu driver requested but Kuzu is not installed."}))
        return None
    try:
        kuzu_root = config.get("kuzu_path", ".")
        if not os.path.exists(kuzu_root): os.makedirs(kuzu_root)
        safe_group = "".join(c for c in group_id if c.isalnum() or c in (' ', '_', '-')).strip() or "default"
        db_path = os.path.join(kuzu_root, safe_group)
        driver = KuzuDriver(db=db_path)
        driver._database = group_id
        try:
            conn = kuzu.Connection(driver.db)
            for query in [
                "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary'])",
                "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact'])"
            ]:
                try: conn.execute(query)
                except Exception: pass
        except Exception as e: print(json.dumps({"error": f"Kuzu Index Patch Failed: {e}"}))
        return driver
    except Exception as e:
        print(json.dumps({"error": f"Kuzu Init Failed: {e}"}))
        return None

# --- CORE LOGIC ---
def generate_insight(config: Dict[str, Any], content: str, mode: str) -> str:
    # (No changes needed in this function)
    if mode not in ANALYSIS_PROMPTS: return ""
    params = get_llm_config_params(config.get("insight_llm", {}))
    if not params["api_key"]: return "[Analysis Error: No API Key for Insight Model]"
    try:
        client = openai.OpenAI(api_key=params["api_key"], base_url=params["base_url"])
        temperature = config.get("advanced", {}).get("insight_model_temperature", 0.3)
        call_args = {
            "model": params["model"],
            "messages": [
                {"role": "system", "content": f"You are a memory optimization engine. {ANALYSIS_PROMPTS[mode]}"},
                {"role": "user", "content": content}
            ],
            "temperature": temperature,
            "max_tokens": 500
        }
        if "google" not in params.get("provider", ""):
            call_args["frequency_penalty"] = 0.0
            call_args["presence_penalty"] = 0.0
        response = client.chat.completions.create(**call_args)
        return response.choices[0].message.content.strip() if response.choices[0].message.content else ""
    except Exception as e:
        return f"[Analysis Failed: {str(e)}]"

async def execute_operation(client: Graphiti, args: argparse.Namespace, config: Dict[str, Any], driver_type: str):
    """Route and execute the requested operation."""
    
    if args.operation == "add":
        original_content = args.content
        
        # 1. Lifecycle Check
        if check_lifecycle(original_content, config):
            print(json.dumps({"status": "skipped", "message": "Content pruned by lifecycle rules."}))
            return

        # 2. Sanitization
        sanitized_content = sanitize_memory(original_content, config)
        if not sanitized_content:
            print(json.dumps({"status": "skipped", "message": "Content was empty after sanitization."}))
            return

        # 3. Ingestion guardrails
        ingestion_reason = evaluate_ingestion_controls(sanitized_content, config)
        if ingestion_reason:
            print(json.dumps({"status": "skipped", "message": ingestion_reason}))
            return

        # 4. Insight Generation (on original content for full context)
        insight = generate_insight(config, original_content, args.mode)

        # 5. Assemble Content
        # Start with the cleaned conversation
        final_content = sanitized_content

        # Prepend insight if it exists
        if insight:
            final_content = f"[MEMORY_MODE: {args.mode}]\n[INSIGHT_START]\n{insight}\n[INSIGHT_END]\n\n{final_content}"

        # 6. Intelligence Layer (e.g., tagging)
        final_content = apply_intelligence_layer(final_content, original_content, config)

        # 7. Custom Memory Tags
        advanced = config.get("advanced", {})
        custom_tags = advanced.get("custom_memory_tags", "")
        if custom_tags:
            tag_list = [t.strip() for t in custom_tags.split(",") if t.strip()]
            tag_prefix = "".join([f"[TAG: {t}]" for t in tag_list])
            final_content = f"{tag_prefix} {final_content}"

        # 8. Add to Database
        await client.add_episode(
            name=args.name,
            episode_body=final_content,
            source=EpisodeType.text,
            source_description=f"Py-GPT Chat ({args.mode})",
            reference_time=datetime.now(),
            group_id=args.group_id
        )
        print(json.dumps({"status": "success", "message": f"Episode added. [Backend: {driver_type}] [DB: {args.group_id}]"}))

    elif args.operation == "search":
        results = await client.search(
            query=args.query,
            num_results=args.limit,
            group_ids=[args.group_id] if args.group_id else None
        )
        output = [getattr(res, 'fact', getattr(res, 'body', getattr(res, 'content', str(res)))) for res in results if res]
        print(json.dumps({"status": "success", "results": output}))

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--name", help="Episode Name")
    parser.add_argument("--content", help="Episode Content")
    parser.add_argument("--query", help="Search Query")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", help="Memory Mode", default="Chatbot")
    parser.add_argument("--group_id", help="Database Name", default="neo4j")
    args = parser.parse_args()
    config = json.loads(args.config)

    # 1. Driver Setup
    driver_type = config.get("driver_type", "Neo4j")
    graph_driver = None
    if driver_type == "Kuzu":
        graph_driver = setup_kuzu_driver(config, args.group_id)
        if not graph_driver: return
    elif driver_type == "Neo4j" and args.group_id:
        setup_neo4j_driver(config, args.group_id)

    # 2. LLM Client & Embedder Setup
    llm_conf = config.get("llm", {})
    llm_params = get_llm_config_params(llm_conf)
    if llm_params["base_url"]: os.environ["OPENAI_BASE_URL"] = llm_params["base_url"]
    else: os.environ.pop("OPENAI_BASE_URL", None)
    if llm_params["api_key"]: os.environ["OPENAI_API_KEY"] = llm_params["api_key"]
    
    try:
        llm_config = LLMConfig(
            model=llm_params["model"], api_key=llm_params["api_key"], base_url=llm_params["base_url"],
            temperature=0.0, max_tokens=int(llm_conf.get("max_tokens", 8192))
        )
        custom_llm = OpenAIGenericClient(llm_config)

        embed_conf = config.get("embedding", {})
        embed_model = embed_conf.get("model", "text-embedding-3-small")
        embed_provider = embed_conf.get("provider", "OpenAI")
        embedder = None
        if embed_provider == "Google":
            if GeminiEmbedder:
                google_key = embed_conf.get("google_api_key") or os.environ.get("GOOGLE_API_KEY")
                if not google_key:
                    print(json.dumps({"error": "Google Embedding selected but no API Key found."}))
                    return
                embedder = GeminiEmbedder(GeminiEmbedderConfig(api_key=google_key, embedding_model=embed_model))
            else:
                print(json.dumps({"error": "Google GenAI library missing."}))
                return
        elif embed_provider == "Ollama":
             embedder = OpenAIEmbedder(OpenAIEmbedderConfig(embedding_model=embed_model, base_url="http://localhost:11434/v1", api_key="ollama"))
        else:
            embedder = OpenAIEmbedder(OpenAIEmbedderConfig(embedding_model=embed_model))

        # 4. Initialize Graphiti
        client_kwargs = {"llm_client": custom_llm, "embedder": embedder}
        if graph_driver:
            client_kwargs["graph_driver"] = graph_driver
        else:
             client_kwargs.update({"uri": config["uri"], "user": config["user"], "password": config["password"]})

        client = Graphiti(**client_kwargs)
        if driver_type == "Neo4j":
            try: await client.build_indices_and_constraints()
            except Exception: pass

        # 5. Execute
        await execute_operation(client, args, config, driver_type)
            
    except Exception as e:
        print(json.dumps({"error": f"Runner Execution Failed: {e}"}))

if __name__ == "__main__":
    asyncio.run(main())
