import asyncio
import json
import sys
import os
import argparse
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Iterable, Match, Tuple

from pydantic import BaseModel, Field, ValidationError

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

# --- EMOTION ENGINE CONSTANTS ---
_WORD_RE = re.compile(r"[a-zA-Z']+")
_ELONG_RE = re.compile(r"([a-zA-Z])\\1{2,}")
_EXCL_RE = re.compile(r"!+")
_Q_RE = re.compile(r"\\?+")
_ELLIPSIS_RE = re.compile(r"\\.\\.\\.+")

NEGATIONS = {"not", "no", "never", "n't", "cannot", "can't", "won't", "don't", "didn't", "isn't", "aren't"}
INTENSIFIERS = {"very", "so", "really", "extremely", "super", "totally", "insanely", "incredibly", "quite"}
DIMINISHERS = {"slightly", "kinda", "kindof", "somewhat", "a_bit", "a_little"}

LEXICON: Dict[str, Dict[str, float]] = {
    "Happy": {
        "happy": 1.0, "glad": 0.9, "love": 0.9, "awesome": 1.2, "great": 1.0, "nice": 0.6,
        "yay": 1.2, "smile": 0.7, "good": 0.5, "wonderful": 1.2
    },
    "Amused": {
        "lol": 1.0, "lmao": 1.2, "haha": 1.0, "funny": 1.0, "hilarious": 1.3, "joke": 0.8
    },
    "Excited": {
        "excited": 1.2, "hype": 1.2, "lets": 0.4, "go": 0.4, "can't_wait": 1.3, "stoked": 1.2
    },
    "Curious": {
        "curious": 1.2, "why": 0.8, "how": 0.6, "what": 0.5, "wonder": 1.0, "maybe": 0.4
    },
    "Thoughtful": {
        "think": 0.7, "consider": 1.0, "reflect": 1.1, "hmm": 0.8, "perhaps": 0.7, "analyze": 0.9
    },
    "Sad": {
        "sad": 1.2, "down": 0.8, "tired": 0.7, "hurt": 0.9, "lonely": 1.1, "sorry": 0.7
    },
    "Angry": {
        "angry": 1.2, "mad": 1.1, "annoyed": 1.0, "hate": 1.1, "pissed": 1.2, "furious": 1.3
    },
    "Anxious": {
        "anxious": 1.2, "worried": 1.1, "nervous": 1.0, "scared": 1.1, "afraid": 1.1, "panic": 1.3
    },
}

EMOJI_HINTS: Dict[str, Dict[str, float]] = {
    "Happy": {"\ud83d\ude0a": 1.0, "\ud83d\ude04": 1.1, "\ud83d\ude01": 1.0, "\ud83d\ude0d": 1.2},
    "Amused": {"\ud83d\ude02": 1.2, "\ud83e\udd23": 1.3},
    "Sad": {"\ud83d\ude13": 0.9, "\ud83d\ude22": 1.2, "\ud83d\ude2d": 1.3},
    "Angry": {"\ud83d\ude21": 1.3, "\ud83e\udd2c": 1.3},
    "Curious": {"\ud83e\udd14": 1.0},
    "Excited": {"\ud83d\udd25": 1.0, "\ud83e\udd29": 1.2},
}

SENSITIVITY = {
    "Low": {"neutral_threshold": 1.6, "signal_scale": 0.9},
    "Medium": {"neutral_threshold": 1.2, "signal_scale": 1.0},
    "High": {"neutral_threshold": 0.9, "signal_scale": 1.1},
}

# --- CONSTANTS ---
ANALYSIS_PROMPTS = {
    "Identity": (
        "You are a graph-memory analyzer. Determine whether the assistant persona changed in voice, boundaries, "
        "stance, capabilities, relationship dynamic, or recurring traits.\n"
        "Extract only durable persona traits and relationship rules.\n\n"
        "OUTPUT: JSON only using the Graph Delta contract.\n"
        "NODES to create/update: PersonaTrait, Boundary, StylePreference, RelationshipDynamic.\n"
        "EDGES: SHIFTS (old->new), REINFORCES, AVOIDS, PREFERS.\n"
        "Include: confidence, salience, ttl_days, and 1-2 evidence snippets.\n"
        "Do NOT store erotic/sexual content, private identifying details, or one-off banter unless it sets a lasting rule."
    ),
    "Assistant": (
        "You are a graph-memory analyzer. Extract implied needs, repeated workflow patterns, automation opportunities, "
        "and stable user preferences for how the assistant should operate.\n\n"
        "OUTPUT: JSON only using the Graph Delta contract.\n"
        "NODES: Preference, WorkflowPattern, Tooling, Project, Constraint.\n"
        "EDGES: PREFERS (User->Preference), REQUESTS (User->WorkflowPattern), DEPENDS_ON (Project->Tooling/Constraint).\n"
        "Gate writes to durable info only; assign ttl_days (180+ for stable preferences, 30 for active projects, 7 for minor habits)."
    ),
    "Chatbot": (
        "You are a graph-memory analyzer. Decide whether this interaction is primarily about User context, Assistant identity, or Mixed.\n"
        "Extract only the minimal nodes/edges needed to preserve continuity (preferences, roles, relationship rules).\n\n"
        "OUTPUT: JSON only using the Graph Delta contract.\n"
        "Also fill: dominant_context with Identity/User/Mixed and a 1-2 sentence summary.\n"
        "If Mixed, create a RelationshipDynamic node and link both User and Assistant persona nodes to it."
    ),
    "Productivity": (
        "You are a graph-memory analyzer. Extract actionable tasks, sub-tasks, dependencies, blockers, and next actions.\n"
        "Represent them as a DAG in the graph.\n\n"
        "OUTPUT: JSON only using the Graph Delta contract.\n"
        "NODES: Task, Project, Dependency, Blocker, Milestone.\n"
        "EDGES: DEPENDS_ON (Task->Task/Dependency), BLOCKED_BY (Task->Blocker), PART_OF (Task->Project).\n"
        "Each Task must include: status (new/ongoing/done), priority (low/med/high), and ttl_days (30 unless stated long-term)."
    ),
    "Research": (
        "You are a graph-memory analyzer. Extract factual claims, hypotheses, and cited sources. Assess reliability and detect contradictions.\n\n"
        "OUTPUT: JSON only using the Graph Delta contract.\n"
        "NODES: Claim, Source, Uncertainty.\n"
        "EDGES: SUPPORTS (Source->Claim), CONTRADICTS (Claim->Claim), QUALIFIES (Uncertainty->Claim).\n"
        "For each Claim include: claim_type (fact/hypothesis/opinion), confidence, and evidence snippet.\n"
        "If no source is provided, create a Source node labeled 'Uncited' with low reliability."
    ),
    "Discourse": (
        "You are a graph-memory analyzer. Map the argument structure: premises, conclusions, warrants, and rhetorical moves.\n"
        "Detect fallacies only if confidence is high.\n\n"
        "OUTPUT: JSON only using the Graph Delta contract.\n"
        "NODES: Premise, Conclusion, RhetoricalMove, Fallacy.\n"
        "EDGES: SUPPORTS (Premise->Conclusion), ATTACKS (Premise->Premise), FRAMES (RhetoricalMove->Conclusion), CONTAINS (Fallacy->Premise/Conclusion).\n"
        "Set ttl_days to 7 unless the argument reflects a stable belief preference."
    ),
    "ResolveEntities": (
        "You are an entity resolution module for a graph memory system.\n"
        "Given the interaction text and a list of existing canonical entities (if provided), produce merge and alias suggestions.\n\n"
        "OUTPUT: JSON only using the Graph Delta contract, but only fill resolution_hints.\n"
        "Rules: prefer canonical forms like 'User:Joshua', 'AssistantPersona:Nova', 'Project:MemoryPlus'.\n"
        "Include confidence for each alias/merge suggestion."
    ),
    "MemoryGate": (
        "Decide if this interaction should write to long-term graph memory.\n"
        "Only write if it contains durable preferences, ongoing projects, stable identity traits, or critical constraints.\n\n"
        "OUTPUT: JSON only using the Graph Delta contract.\n"
        "If should_write_memory is false, leave graph_delta empty and explain why in summary."
    ),
}

class InsightGraphDelta(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    resolution_hints: List[Dict[str, Any]] = Field(default_factory=list)


class InsightPayload(BaseModel):
    summary: Optional[str] = None
    dominant_context: Optional[str] = None
    should_write_memory: Optional[bool] = None
    graph_delta: InsightGraphDelta = Field(default_factory=InsightGraphDelta)


def _validated_insight_json(raw_text: str) -> str:
    parsed = json.loads(raw_text)
    if isinstance(parsed, list):
        parsed = {"graph_delta": {"nodes": parsed}}
    payload = InsightPayload.parse_obj(parsed)
    return json.dumps(payload.dict(), indent=2)

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
        preserve_tagged = sanitization_config.get("preserve_tagged_code", True)

        def _replace_code(match: re.Match) -> str:
            block_body = match.group(1) or ""
            if preserve_tagged and "[KEEP_CODE]" in block_body:
                cleaned = block_body.replace("[KEEP_CODE]", "").strip()
                if cleaned:
                    return f"\n[CODE_SNIPPET]\n{cleaned}\n[/CODE_SNIPPET]\n"
                return ""
            return ""

        text = re.sub(r'```(.*?)```', _replace_code, text, flags=re.DOTALL)
    
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

def _normalize_tokens(tokens: Iterable[str]) -> Iterable[str]:
    for t in tokens:
        yield t.lower()


def _negation_window(tokens: List[str], idx: int, window: int = 3) -> bool:
    start = max(0, idx - window)
    return any(tok in NEGATIONS for tok in tokens[start:idx])


def _intensity_multiplier(tokens: List[str], idx: int, window: int = 2) -> float:
    start = max(0, idx - window)
    mult = 1.0
    for tok in tokens[start:idx]:
        if tok in INTENSIFIERS:
            mult *= 1.25
        elif tok in DIMINISHERS:
            mult *= 0.8
    return mult


def detect_emotion(text: str, sensitivity: str = "Medium") -> Tuple[str, float]:
    cfg = SENSITIVITY.get(sensitivity, SENSITIVITY["Medium"])
    scale = cfg["signal_scale"]

    raw_tokens = _WORD_RE.findall(text)
    tokens = list(_normalize_tokens(raw_tokens))
    scores: Dict[str, float] = {k: 0.0 for k in LEXICON.keys()}

    for i, tok in enumerate(tokens):
        for emotion, vocab in LEXICON.items():
            base = vocab.get(tok, 0.0)
            if base <= 0:
                continue
            mult = _intensity_multiplier(tokens, i)
            if _negation_window(tokens, i):
                base *= -0.7
            scores[emotion] += base * mult

    for ch in text:
        for emotion, hints in EMOJI_HINTS.items():
            if ch in hints:
                scores[emotion] += hints[ch] * scale

    excls = len(_EXCL_RE.findall(text))
    qs = len(_Q_RE.findall(text))
    ell = len(_ELLIPSIS_RE.findall(text))
    elong = len(_ELONG_RE.findall(text))
    alpha_chars = sum(1 for c in text if c.isalpha())
    caps_chars = sum(1 for c in text if c.isupper())
    caps_ratio = caps_chars / max(1, alpha_chars)

    scores["Excited"] += (0.5 * excls + 0.2 * elong) * scale
    scores["Curious"] += (0.6 * qs) * scale
    scores["Thoughtful"] += (0.4 * ell) * scale
    if caps_ratio > 0.25:
        scores["Angry"] += (0.4 * excls) * scale

    best_emotion, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score <= 0 or best_score < cfg["neutral_threshold"]:
        return "Neutral", 0.0
    total_positive = sum(max(val, 0.0) for val in scores.values())
    confidence = best_score / max(total_positive, 1e-6)
    return best_emotion, min(1.0, confidence)

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
        emotion, confidence = detect_emotion(original_content, sensitivity)
        if confidence < 0.35:
            llm_emotion = _classify_emotion_via_llm(original_content, config)
            if llm_emotion:
                emotion = llm_emotion
        content = f"[EMOTION: {emotion}] {content}"

    # Topic Tagging
    if intelligence_config.get("enable_topic_tagging", False):
        topics = detect_topics(original_content)
        if topics == ["Unclassified"]:
            llm_topics = _classify_topics_via_llm(original_content, config)
            if llm_topics:
                topics = llm_topics
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


def _build_support_llm_client(config: Dict[str, Any]):
    llm_conf = config.get("llm", {})
    params = get_llm_config_params(llm_conf)
    api_key = params.get("api_key")
    if not api_key:
        return None, params
    client = openai.OpenAI(api_key=api_key, base_url=params.get("base_url"))
    return client, params


def _classify_emotion_via_llm(text: str, config: Dict[str, Any]) -> Optional[str]:
    try:
        client, params = _build_support_llm_client(config)
    except Exception:
        return None
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model=params.get("model"),
            messages=[
                {"role": "system", "content": "Classify the user's emotional tone. Respond with a single concise emotion label such as Happy, Sad, Angry, Anxious, Excited, or Neutral."},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=12,
        )
        message = resp.choices[0].message.content if resp.choices else ""
        if not message:
            return None
        label = re.split(r"[\\n,]", message.strip())[0]
        return label.title()
    except Exception:
        return None


def _classify_topics_via_llm(text: str, config: Dict[str, Any]) -> Optional[List[str]]:
    try:
        client, params = _build_support_llm_client(config)
    except Exception:
        return None
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model=params.get("model"),
            messages=[
                {"role": "system", "content": "List up to three short topic tags (1-2 words each) that describe the user's message. Respond as a comma-separated list, e.g., 'Python, AsyncIO, Testing'."},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=30,
        )
        content_resp = resp.choices[0].message.content if resp.choices else ""
        if not content_resp:
            return None
        topics = [part.strip() for part in content_resp.split(",") if part.strip()]
        normalized = []
        for topic in topics:
            clean = re.sub(r"[^a-zA-Z0-9 _-]", "", topic).strip()
            if clean:
                normalized.append(clean)
        return normalized or None
    except Exception:
        return None

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
    advanced = config.get("advanced", {})
    if mode == "CustomPrompt":
        system_prompt = advanced.get("custom_analysis_prompt", "").strip()
    else:
        system_prompt = ANALYSIS_PROMPTS.get(mode)
    if not system_prompt:
        return ""
    params = get_llm_config_params(config.get("insight_llm", {}))
    if not params["api_key"]: return "[Analysis Error: No API Key for Insight Model]"
    try:
        client = openai.OpenAI(api_key=params["api_key"], base_url=params["base_url"])
        temperature = advanced.get("insight_model_temperature", 0.3)
        base_instruction = "You are a memory optimization engine."
        if base_instruction not in system_prompt:
            full_prompt = f"{base_instruction} {system_prompt}".strip()
        else:
            full_prompt = system_prompt
        base_messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": content},
        ]
        last_response_text = ""
        error_msg = ""
        for attempt in range(2):
            messages = list(base_messages)
            if attempt > 0:
                messages.append({
                    "role": "user",
                    "content": f"Your previous response could not be parsed ({error_msg}). "
                               "Respond with valid JSON only that matches the Graph Delta contract."
                })
            call_args = {
                "model": params["model"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 500
            }
            if "google" not in params.get("provider", ""):
                call_args["frequency_penalty"] = 0.0
                call_args["presence_penalty"] = 0.0
            response = client.chat.completions.create(**call_args)
            message = response.choices[0].message.content if response.choices else ""
            last_response_text = (message or "").strip()
            if not last_response_text:
                continue
            try:
                return _validated_insight_json(last_response_text)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                error_msg = str(exc)
                continue
        if last_response_text:
            return json.dumps({"raw_insight": last_response_text}, indent=2)
        if error_msg:
            return f"[Analysis Failed: {error_msg}]"
        return "[Analysis Failed: empty insight response]"
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

        # 3. Insight Generation (on original content for full context)
        insight_str = generate_insight(config, original_content, args.mode)

        # FIX: Ensure insight is a mapping, not a list
        insight = None
        if insight_str and insight_str.strip().startswith(('{', '[')):
            try:
                parsed_insight = json.loads(insight_str)
                if isinstance(parsed_insight, list):
                    # If the top-level element is a list, wrap it in a dict.
                    insight = {"extracted_entities": parsed_insight}
                else:
                    # It's already a dict, or some other valid JSON non-list type
                    insight = parsed_insight
            except json.JSONDecodeError:
                # If it's not valid JSON, treat it as a simple string.
                insight = {"raw_insight": insight_str}
        else:
            # Handle cases where insight is not JSON or is empty
            insight = {"raw_insight": insight_str}

        # 4. Assemble Content
        # Start with the cleaned conversation
        final_content = sanitized_content

        # Prepend insight if it exists and is not empty
        if insight and (insight.get("extracted_entities") or insight.get("raw_insight")):
            # Convert the insight object back to a JSON string for storing
            insight_json_str = json.dumps(insight, indent=2)
            final_content = f"[MEMORY_MODE: {args.mode}]\\n[INSIGHT_START]\\n{insight_json_str}\\n[INSIGHT_END]\\n\\n{final_content}"

        # 5. Intelligence Layer (e.g., tagging)
        final_content = apply_intelligence_layer(final_content, original_content, config)

        # 6. Custom Memory Tags
        advanced = config.get("advanced", {})
        custom_tags = advanced.get("custom_memory_tags", "")
        if custom_tags:
            tag_list = [t.strip() for t in custom_tags.split(",") if t.strip()]
            tag_prefix = "".join([f"[TAG: {t}]" for t in tag_list])
            final_content = f"{tag_prefix} {final_content}"

        # 7. Add to Database
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

        embed_conf = config.get("embedding", {}) # Embedding configuration
        embed_model = embed_conf.get("model", "mxbai-embed-large")
        embed_provider = embed_conf.get("provider", "Ollama")
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
            if embed_model == "mxbai-embed-large":
                embed_model = "mxbai-embed-large:latest" # Force :latest for Ollama
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
