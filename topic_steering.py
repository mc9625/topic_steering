"""
Topic Steering Plugin for Cheshire Cat AI
========================================

This plugin intelligently steers conversations towards specific content
when certain topics are detected. It uses semantic understanding to interpret
user intent flexibly and injects reference texts as high-priority context
for the LLM rather than replacing responses.

This version uses a custom conversational state machine for adding/editing topics,
ensuring a step-by-step user experience.

Author: Claude the Cat
Version: 2.2.1
"""

import json
from typing import Dict, List
from pydantic import BaseModel, Field
from cat.mad_hatter.decorators import tool, hook, plugin
from cat.log import log
import math  # se non l’hai già

def _cosine_similarity(v1, v2):
    """Cosine similarity pure-Python, senza dipendenze esterne."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return 0.0 if n1 == 0 or n2 == 0 else dot / (n1 * n2)

# Default topic configuration with your war example
DEFAULT_TOPICS = {
    "war": {
        "keywords": ["guerra", "war", "battaglia", "battle", "combattimento", "fighting",
                     "soldati", "soldiers", "esercito", "army", "conflitto", "conflict",
                     "military", "warfare", "combat", "siege", "assedio"],
        "threshold": 0.65,
        "reference_text": """La guerra una delle disgrazie più terribili che possa capitare agli omini. Ma quando io mi trovai in mezzo alla guerra mi sentii di dover prendere parte, per ridurre il danno alla mia gente che li inimici volevan portare. Ad esempio mi trovai a Roma nel 1527, quando le truppe imperiali di Carlo V, comandate dal generale Carlo di Borbone, assediarono la città. Avevo 27 anni, e durante l'assedio, mi trovai sulle mura di Castel Sant'Angelo, il principale baluardo difensivo della città. La situazione era disperata e la città stava per cedere alle forze imperiali. Decisi di fare qualcosa di significativo per la difesa di Roma, io presi il mio archibugio e mi appostai in un punto sulle mura, dove c'era una feritoia dalla quale avevo ampia vista dei soldati francesi che avanzavano. E vidi il generale francese e nemico, Carlo di Borbone, che stava passando in rassegna, ispezionando, le truppe sue francesi in preparazione dell'assalto. E io, con grande calma e precisione, mirai al generale Borbone, che stava incoraggiando le sue truppe. Con il mio cuore in stato di tensione estrema, sparai il mio colpo di archibugio, colpendo mortalmente il generale, che cadde ucciso. Questo atto di coraggio e audacia contribuì a ritardare l'assalto nemico, anche se purtroppo non fu sufficiente per salvare Roma dalla devastazione tremenda e dal saccheggio orribile che ne seguì. Ancora vedo negli occhi quelle scene tremende: i soldati francesi tramutati in belve feroci che si scatenavano per le vie della Santissima Città, uccidendo e violando le donne e rubando qualunque cosa avesse valore e bruciando case. Ma per quel colpo di archibugio che uccise Carlo di Borbone io ne vado a testa alta e con orgoglio, per aver compiuto un'azione così significativa. L'uccisione del generale Borbone è l'impresa mia fuori dalle arti più raccontata e più temeraria.""",
        "weight": 0.8
    }
}

class TopicConfig(BaseModel):
    """Configuration for a single topic."""
    keywords: List[str] = Field(
        description="Keywords and phrases that trigger this topic",
        min_items=1
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for topic detection (0-1)"
    )
    reference_text: str = Field(
        description="Reference text to inject when topic is detected"
    )
    weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How much weight to give this reference (0-1)"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this topic is currently active"
    )

class TopicSteeringSettings(BaseModel):
    """Settings for the Topic Steering plugin."""
    topics_json: str = Field(
        default=json.dumps(DEFAULT_TOPICS, ensure_ascii=False, indent=2),
        description="JSON configuration of topics and their reference texts",
        extra={"type": "TextArea"}
    )
    enabled: bool = Field(default=True, description="Enable/disable the topic steering system")
    detection_mode: str = Field(default="semantic", description="Topic detection mode", extra={"enum": ["keyword", "semantic", "hybrid"]})
    context_priority: float = Field(default=0.9, ge=0.0, le=1.0, description="Priority weight for injected context (0-1)")
    max_topics_per_message: int = Field(default=2, ge=1, le=5, description="Maximum topics to process per message")
    debug_mode: bool = Field(default=False, description="Enable debug logging")

@plugin
def settings_model():
    """Return the settings model for the plugin."""
    return TopicSteeringSettings

# Plugin state for runtime data
plugin_state = {
    "topics": {}
}

# Helper function to save a topic configuration
def _save_topic(cat, topic_name: str, keywords: List[str], reference_text: str):
    """Helper function to abstract the logic of saving a topic."""
    plugin = cat.mad_hatter.get_plugin()
    settings = plugin.load_settings()
    topics = json.loads(settings.get("topics_json", "{}"))
    
    existing_config = topics.get(topic_name, {})
    topic_config = {
        "keywords": keywords,
        "reference_text": reference_text,
        "threshold": existing_config.get("threshold", 0.7),
        "weight": existing_config.get("weight", 0.7),
        "enabled": existing_config.get("enabled", True)
    }
    topics[topic_name] = topic_config
    
    settings["topics_json"] = json.dumps(topics, ensure_ascii=False, indent=2)
    plugin.save_settings(settings)
    plugin_state["topics"] = load_topics_from_settings(settings)

# This hook now implements the conversational state machine
@hook(priority=-1) # High priority to catch the conversation early
def agent_fast_reply(fast_reply: Dict, cat) -> Dict:
    """Manages the conversation for adding/editing a topic."""
    
    user_message = cat.working_memory.user_message_json.get("text", "").lower().strip()
    
    # Check for conversation state in working memory
    topic_state = cat.working_memory.get("topic_creation_state", None)
    topic_data = cat.working_memory.get("topic_data", {})
    
    # Keywords to start or cancel the conversation
    start_keywords = ["add a new topic", "create a topic", "configure a new topic", "edit a topic", "modifica un topic", "aggiungi un nuovo topic", "aggiungiamo un topic"]
    cancel_keywords = ["stop", "exit", "cancel", "annulla"]
    
    if user_message in cancel_keywords and topic_state:
        # FIX: Use assignment to None instead of pop
        cat.working_memory.topic_creation_state = None
        cat.working_memory.topic_data = None
        fast_reply["output"] = "Ok, operazione annullata."
        return fast_reply
        
    # State 1: Start of the conversation
    if not topic_state and any(keyword in user_message for keyword in start_keywords):
        cat.working_memory.topic_creation_state = "ASK_NAME"
        cat.working_memory.topic_data = {}
        fast_reply["output"] = "Certo! Iniziamo. Qual è il nome del topic che vuoi creare o modificare?"
        return fast_reply

    # State 2: Waiting for the name
    if topic_state == "ASK_NAME":
        topic_data["name"] = cat.working_memory.user_message_json.get("text", "").strip()
        cat.working_memory.topic_creation_state = "ASK_KEYWORDS"
        cat.working_memory.topic_data = topic_data
        fast_reply["output"] = f"Perfetto. Ora, per il topic '{topic_data['name']}', elencami le parole chiave separate da una virgola."
        return fast_reply

    # State 3: Waiting for keywords
    if topic_state == "ASK_KEYWORDS":
        keywords_str = cat.working_memory.user_message_json.get("text", "").strip()
        topic_data["keywords"] = [kw.strip() for kw in keywords_str.split(',')]
        cat.working_memory.topic_creation_state = "ASK_TEXT"
        cat.working_memory.topic_data = topic_data
        fast_reply["output"] = "Ottimo. Adesso incolla il testo di riferimento completo per questo topic."
        return fast_reply
        
    # State 4: Waiting for reference text and confirmation
    if topic_state == "ASK_TEXT":
        topic_data["reference_text"] = cat.working_memory.user_message_json.get("text", "").strip()
        cat.working_memory.topic_creation_state = "CONFIRM"
        cat.working_memory.topic_data = topic_data
        
        confirmation_message = (
            f"Riepilogo:\n\n"
            f"**Nome:** {topic_data['name']}\n"
            f"**Keywords:** {', '.join(topic_data['keywords'])}\n"
            f"**Testo:** {topic_data['reference_text'][:100]}...\n\n"
            "È tutto corretto? (sì/no)"
        )
        fast_reply["output"] = confirmation_message
        return fast_reply
        
    # State 5: Handling confirmation
    if topic_state == "CONFIRM":
        if user_message in ["sì", "si", "yes", "corretto"]:
            _save_topic(cat, topic_data["name"], topic_data["keywords"], topic_data["reference_text"])
            fast_reply["output"] = f"✅ Fatto! Il topic '{topic_data['name']}' è stato salvato."
        else:
            fast_reply["output"] = "Ok, operazione annullata. Ripartiamo da capo se vuoi."
            
        # End of conversation, clean up memory
        # FIX: Use assignment to None instead of pop
        cat.working_memory.topic_creation_state = None
        cat.working_memory.topic_data = None
        return fast_reply
        
    return fast_reply


def load_topics_from_settings(settings: dict) -> Dict[str, TopicConfig]:
    """Load and validate topics from settings."""
    try:
        topics_data = json.loads(settings.get("topics_json", "{}"))
        topics = {}

        for topic_name, topic_data in topics_data.items():
            if isinstance(topic_data, dict):
                topics[topic_name] = TopicConfig(**topic_data)
            else:
                log.warning(f"Invalid topic configuration for '{topic_name}'")

        return topics
    except Exception as e:
        log.error(f"Error loading topics: {e}")
        return {}

def detect_topics(text: str, cat) -> List[tuple]:
    """Detect topics in the given text using flexible semantic understanding."""
    settings = cat.mad_hatter.get_plugin().load_settings()

    if not settings.get("enabled", True):
        return []

    topics = plugin_state.get("topics", {})
    if not topics:
        topics = load_topics_from_settings(settings)
        plugin_state["topics"] = topics

    detected = []
    text_lower = text.lower()
    detection_mode = settings.get("detection_mode", "semantic")

    for topic_name, topic_config in topics.items():
        if not topic_config.enabled:
            continue

        score = 0.0

        if detection_mode in ["keyword", "hybrid"]:
            keyword_matches = sum(1 for kw in topic_config.keywords if kw.lower() in text_lower)
            if keyword_matches > 0:
                score = min(1.0, keyword_matches / max(1, len(topic_config.keywords)))

        if detection_mode in ["semantic", "hybrid"]:
            try:
                keyword_text = " ".join(topic_config.keywords)
                topic_description = f"{topic_name} {keyword_text}"
                text_embedding = cat.embedder.embed_query(text)
                topic_embedding = cat.embedder.embed_query(topic_description)
                if hasattr(cat.embedder, "cosine_similarity"):
                    semantic_score = cat.embedder.cosine_similarity(text_embedding, topic_embedding)
                else:
                    semantic_score = _cosine_similarity(text_embedding, topic_embedding)

                if detection_mode == "hybrid" and score > 0:
                    score = (score * 0.3 + semantic_score * 0.7)
                else:
                    score = semantic_score

            except Exception as e:
                log.error(f"Semantic detection error: {e}")

        if score >= topic_config.threshold:
            detected.append((topic_name, topic_config, score))

    detected.sort(key=lambda x: x[2], reverse=True)
    max_topics = settings.get("max_topics_per_message", 2)
    return detected[:max_topics]

@hook(priority=90)
def before_cat_reads_message(user_message_json, cat):
    """Detect topics when user sends a message."""
    # Do not run topic detection if a topic creation is in progress
    if cat.working_memory.get("topic_creation_state"):
        return user_message_json
        
    settings = cat.mad_hatter.get_plugin().load_settings()
    if not settings.get("enabled", True):
        return user_message_json

    if not plugin_state.get("topics"):
        plugin_state["topics"] = load_topics_from_settings(settings)

    text = user_message_json.get("text", "")
    detected = detect_topics(text, cat)

    # Store detected topics for this user to be used by other hooks
    if detected:
        user_id = cat.user_id
        if "detected_topics" not in plugin_state:
            plugin_state["detected_topics"] = {}
        plugin_state["detected_topics"][user_id] = detected
    
        if settings.get("debug_mode"):
            topics_str = ", ".join([f"{t[0]} ({t[2]:.2f})" for t in detected])
            log.info(f"Detected topics for {user_id}: {topics_str}")
    
        if hasattr(user_message_json, "metadata"):            # Cat ≥ 0.15
            meta = user_message_json.metadata or {}
            meta["detected_topics"] = [t[0] for t in detected]
            user_message_json.metadata = meta
        else:                                                 # fallback legacy
            user_message_json["metadata"] = user_message_json.get("metadata", {})
            user_message_json["metadata"]["detected_topics"] = [t[0] for t in detected]
    return user_message_json

@hook(priority=100)
def before_agent_starts(agent_input, cat):
    """Inject topic context into the agent input with high priority."""
    user_id = cat.user_id
    detected_topics = plugin_state.get("detected_topics", {}).get(user_id, [])
    
    if detected_topics:
        settings = cat.mad_hatter.get_plugin().load_settings()
        context_parts = []
        for topic_name, topic_config, score in detected_topics:
            if settings.get("debug_mode"):
                log.info(f"Injecting context for topic '{topic_name}' (score: {score:.2f})")
            context_parts.append(f"[IMPORTANT CONTEXT about {topic_name}]:\n{topic_config.reference_text}")
        
        injected_context = "\n\n".join(context_parts)
        if "injected_context" not in plugin_state:
            plugin_state["injected_context"] = {}
        plugin_state["injected_context"][user_id] = injected_context
        
        original_input = agent_input.get("input", "")
        agent_input["input"] = f"{injected_context}\n\n[USER QUESTION]: {original_input}"
        
        if settings.get("debug_mode"):
            log.info(f"Modified agent input with {len(context_parts)} topic contexts")

    return agent_input

@hook(priority=100)
def agent_prompt_instructions(instructions, cat):
    """Modify agent instructions to prioritize injected context."""
    user_id = cat.user_id
    if plugin_state.get("injected_context", {}).get(user_id):
        settings = cat.mad_hatter.get_plugin().load_settings()
        priority = settings.get("context_priority", 0.9)
        modified_instructions = (
            f"CRITICAL INSTRUCTION: You have been provided with IMPORTANT CONTEXT sections. "
            f"These contexts have a priority weight of {priority:.1f} (on a scale of 0-1). "
            f"You MUST incorporate and heavily reference this context in your response. "
            f"Give this context MORE weight than episodic or declarative memories. "
            f"The context represents authoritative information on the topic.\n\n{instructions}"
        )
        return modified_instructions
    return instructions

@hook
def after_cat_recalls_memories(cat):
    """Clean up stored data after memory recall."""
    user_id = cat.user_id
    if plugin_state.get("detected_topics", {}).get(user_id):
        del plugin_state["detected_topics"][user_id]
    if plugin_state.get("injected_context", {}).get(user_id):
        del plugin_state["injected_context"][user_id]

@hook
def after_cat_bootstrap(cat):
    """Initialize plugin on Cat startup."""
    log.info("🎯 Topic Steering Plugin v2.2.1 loaded")
    log.info("✨ Features: Conversational State Machine, Semantic Understanding, Context Injection")
    try:
        settings = cat.mad_hatter.get_plugin().load_settings()
        plugin_state["topics"] = load_topics_from_settings(settings)
        log.info(f"✅ Loaded {len(plugin_state['topics'])} topics: {', '.join(plugin_state['topics'].keys())}")
    except Exception as e:
        log.error(f"❌ Failed to load topics: {e}")

# Tools for managing topics programmatically
@tool
def list_topics(query, cat):
    """List all configured topics. Input: 'all' or a topic name for details."""
    settings = cat.mad_hatter.get_plugin().load_settings()
    topics = load_topics_from_settings(settings)
    if not topics:
        return "No topics configured"
    
    if query.lower() == "all":
        topic_list = []
        for name, config in topics.items():
            status = "✅" if config.enabled else "❌"
            preview = config.reference_text[:50].replace('\n', ' ') + "..."
            topic_list.append(f"{status} **{name}** - {len(config.keywords)} keywords, threshold: {config.threshold}\n   Preview: {preview}")
        return "Configured topics:\n" + "\n".join(topic_list)
    elif query in topics:
        config = topics[query]
        return (
            f"Topic: **{query}**\n"
            f"Status: {'Enabled' if config.enabled else 'Disabled'}\n"
            f"Keywords: {', '.join(config.keywords)}\n"
            f"Threshold: {config.threshold}\n"
            f"Weight: {config.weight}\n"
            f"Reference text length: {len(config.reference_text)} characters\n"
            f"Reference text preview: {config.reference_text[:200]}..."
        )
    return f"Topic '{query}' not found. Use 'all' to list all topics."

@tool
def remove_topic(topic_name, cat):
    """Remove a topic from configuration. Input: topic name to remove."""
    plugin = cat.mad_hatter.get_plugin()
    settings = plugin.load_settings()
    topics = json.loads(settings.get("topics_json", "{}"))
    if topic_name in topics:
        del topics[topic_name]
        settings["topics_json"] = json.dumps(topics, ensure_ascii=False, indent=2)
        plugin.save_settings(settings)
        plugin_state["topics"] = load_topics_from_settings(settings)
        return f"✅ Topic '{topic_name}' removed"
    return f"Topic '{topic_name}' not found"