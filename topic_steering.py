"""
Topic Steering Plugin for Cheshire Cat AI
========================================

This plugin intelligently steers conversations towards specific content
when certain topics are detected. It uses semantic understanding to interpret
user intent flexibly and injects reference texts as high-priority context
for the LLM rather than replacing responses.

Author: Claude the Cat
Version: 2.0.0
"""

import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from cat.mad_hatter.decorators import tool, hook, plugin
from cat.log import log

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
    
    # JSON string containing topics configuration
    topics_json: str = Field(
        default=json.dumps(DEFAULT_TOPICS, ensure_ascii=False, indent=2),
        description="JSON configuration of topics and their reference texts",
        extra={"type": "TextArea"}
    )
    
    # Global settings
    enabled: bool = Field(
        default=True,
        description="Enable/disable the topic steering system"
    )
    
    detection_mode: str = Field(
        default="semantic",
        description="Topic detection mode",
        extra={"enum": ["keyword", "semantic", "hybrid"]}
    )
    
    context_priority: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Priority weight for injected context (0-1)"
    )
    
    max_topics_per_message: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum topics to process per message"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Enable debug logging"
    )

@plugin
def settings_model():
    """Return the settings model for the plugin."""
    return TopicSteeringSettings

# Plugin state for runtime data
plugin_state = {
    "topics": {},
    "detected_topics": {},  # Store per user_id
    "injected_context": {}  # Store injected context per user_id
}

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
        
        # Keyword detection
        if detection_mode in ["keyword", "hybrid"]:
            keyword_matches = sum(1 for kw in topic_config.keywords if kw.lower() in text_lower)
            if keyword_matches > 0:
                score = min(1.0, keyword_matches / max(1, len(topic_config.keywords)))
        
        # Semantic detection - more flexible interpretation
        if detection_mode in ["semantic", "hybrid"]:
            try:
                # Create a richer query that captures the topic essence
                keyword_text = " ".join(topic_config.keywords)
                topic_description = f"{topic_name} {keyword_text}"
                
                # Get embeddings
                text_embedding = cat.embedder.embed_query(text)
                topic_embedding = cat.embedder.embed_query(topic_description)
                
                # Calculate semantic similarity
                semantic_score = cat.embedder.cosine_similarity(text_embedding, topic_embedding)
                
                if detection_mode == "hybrid" and score > 0:
                    # Combine scores with more weight on semantic
                    score = (score * 0.3 + semantic_score * 0.7)
                else:
                    score = semantic_score
                    
            except Exception as e:
                log.error(f"Semantic detection error: {e}")
        
        if score >= topic_config.threshold:
            detected.append((topic_name, topic_config, score))
    
    # Sort by score and limit
    detected.sort(key=lambda x: x[2], reverse=True)
    max_topics = settings.get("max_topics_per_message", 2)
    
    return detected[:max_topics]

@hook(priority=90)
def before_cat_reads_message(user_message_json, cat):
    """Detect topics when user sends a message."""
    settings = cat.mad_hatter.get_plugin().load_settings()
    
    if not settings.get("enabled", True):
        return user_message_json
    
    # Reload topics if needed
    if not plugin_state.get("topics"):
        plugin_state["topics"] = load_topics_from_settings(settings)
    
    text = user_message_json.get("text", "")
    detected = detect_topics(text, cat)
    
    # Store detected topics for this user
    user_id = cat.user_id
    plugin_state["detected_topics"][user_id] = detected
    
    if detected and settings.get("debug_mode"):
        topics_str = ", ".join([f"{t[0]} ({t[2]:.2f})" for t in detected])
        log.info(f"Detected topics for {user_id}: {topics_str}")
    
    # Add detected topics to metadata
    if detected:
        user_message_json["metadata"] = user_message_json.get("metadata", {})
        user_message_json["metadata"]["detected_topics"] = [t[0] for t in detected]
    
    return user_message_json

@hook(priority=100)
def before_agent_starts(agent_input, cat):
    """Inject topic context into the agent input with high priority."""
    settings = cat.mad_hatter.get_plugin().load_settings()
    
    if not settings.get("enabled", True):
        return agent_input
    
    user_id = cat.user_id
    detected_topics = plugin_state.get("detected_topics", {}).get(user_id, [])
    
    if detected_topics:
        # Build the context to inject
        context_parts = []
        
        for topic_name, topic_config, score in detected_topics:
            if settings.get("debug_mode"):
                log.info(f"Injecting context for topic '{topic_name}' (score: {score:.2f})")
            
            context_parts.append(f"[IMPORTANT CONTEXT about {topic_name}]:\n{topic_config.reference_text}")
        
        # Store the injected context
        injected_context = "\n\n".join(context_parts)
        plugin_state["injected_context"][user_id] = injected_context
        
        # Modify the agent input to include the context
        original_input = agent_input.get("input", "")
        agent_input["input"] = f"{injected_context}\n\n[USER QUESTION]: {original_input}"
        
        if settings.get("debug_mode"):
            log.info(f"Modified agent input with {len(context_parts)} topic contexts")
    
    return agent_input

@hook(priority=100)
def agent_prompt_instructions(instructions, cat):
    """Modify agent instructions to prioritize injected context."""
    settings = cat.mad_hatter.get_plugin().load_settings()
    
    if not settings.get("enabled", True):
        return instructions
    
    user_id = cat.user_id
    if user_id in plugin_state.get("injected_context", {}):
        priority = settings.get("context_priority", 0.9)
        
        # Add instructions to prioritize the injected context
        modified_instructions = f"""CRITICAL INSTRUCTION: You have been provided with IMPORTANT CONTEXT sections. 
These contexts have a priority weight of {priority:.1f} (on a scale of 0-1).
You MUST incorporate and heavily reference this context in your response.
Give this context MORE weight than episodic or declarative memories.
The context represents authoritative information on the topic.

{instructions}"""
        
        return modified_instructions
    
    return instructions

@hook
def after_cat_recalls_memories(cat):
    """Clean up stored data after memory recall."""
    user_id = cat.user_id
    
    # Clean up detected topics after use
    if user_id in plugin_state.get("detected_topics", {}):
        del plugin_state["detected_topics"][user_id]
    
    # Clean up injected context after use
    if user_id in plugin_state.get("injected_context", {}):
        del plugin_state["injected_context"][user_id]

@hook
def after_cat_bootstrap(cat):
    """Initialize plugin on Cat startup."""
    log.info("🎯 Topic Steering Plugin v2.0 loaded")
    log.info("✨ Features: Semantic understanding, Context injection, File loading")
    
    # Load initial topics
    try:
        settings = cat.mad_hatter.get_plugin().load_settings()
        topics = load_topics_from_settings(settings)
        plugin_state["topics"] = topics
        
        log.info(f"✅ Loaded {len(topics)} topics: {', '.join(topics.keys())}")
    except Exception as e:
        log.error(f"❌ Failed to load topics: {e}")

@tool
def add_topic(topic_data, cat):
    """Add or update a topic configuration. Input should be JSON with: name, keywords, reference_text, threshold (optional), weight (optional)."""
    try:
        data = json.loads(topic_data)
        
        # Validate required fields
        if not all(k in data for k in ["name", "keywords", "reference_text"]):
            return "Error: Missing required fields. Need: name, keywords, reference_text"
        
        # Load current settings
        plugin = cat.mad_hatter.get_plugin()
        settings = plugin.load_settings()
        topics = json.loads(settings.get("topics_json", "{}"))
        
        # Create topic config
        topic_config = {
            "keywords": data["keywords"] if isinstance(data["keywords"], list) else [data["keywords"]],
            "reference_text": data["reference_text"],
            "threshold": data.get("threshold", 0.7),
            "weight": data.get("weight", 0.7),
            "enabled": data.get("enabled", True)
        }
        
        # Update topics
        topics[data["name"]] = topic_config
        
        # Save settings
        settings["topics_json"] = json.dumps(topics, ensure_ascii=False, indent=2)
        plugin.save_settings(settings)
        
        # Reload topics in runtime
        plugin_state["topics"] = load_topics_from_settings(settings)
        
        return f"✅ Topic '{data['name']}' added/updated successfully with {len(topic_config['keywords'])} keywords"
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def load_topic_from_file(file_data, cat):
    """Load a topic reference text from a file. Input: JSON with 'topic_name' and 'file_path' keys."""
    try:
        data = json.loads(file_data)
        
        if not all(k in data for k in ["topic_name", "file_path"]):
            return "Error: Need both 'topic_name' and 'file_path' in JSON"
        
        topic_name = data["topic_name"]
        file_path = data["file_path"]
        
        # Try to read the file
        try:
            # Use the window.fs.readFile API if in browser context
            if hasattr(cat, 'send_ws_message'):
                # We're in a web context
                import os
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reference_text = f.read()
                else:
                    return f"Error: File not found: {file_path}"
            else:
                # Direct file reading
                with open(file_path, 'r', encoding='utf-8') as f:
                    reference_text = f.read()
                    
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
        # Load current settings
        plugin = cat.mad_hatter.get_plugin()
        settings = plugin.load_settings()
        topics = json.loads(settings.get("topics_json", "{}"))
        
        # Check if topic exists
        if topic_name not in topics:
            return f"Error: Topic '{topic_name}' not found. Create it first with add_topic."
        
        # Update the reference text
        topics[topic_name]["reference_text"] = reference_text
        
        # Save settings
        settings["topics_json"] = json.dumps(topics, ensure_ascii=False, indent=2)
        plugin.save_settings(settings)
        
        # Reload topics
        plugin_state["topics"] = load_topics_from_settings(settings)
        
        char_count = len(reference_text)
        return f"✅ Loaded {char_count} characters from '{file_path}' into topic '{topic_name}'"
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format. Use: {\"topic_name\": \"name\", \"file_path\": \"path/to/file.txt\"}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def list_topics(query, cat):
    """List all configured topics. Input: 'all' or a topic name for details."""
    settings = cat.mad_hatter.get_plugin().load_settings()
    topics = load_topics_from_settings(settings)
    
    if query.lower() == "all":
        if not topics:
            return "No topics configured"
        
        topic_list = []
        for name, config in topics.items():
            status = "✅" if config.enabled else "❌"
            preview = config.reference_text[:50] + "..." if len(config.reference_text) > 50 else config.reference_text
            topic_list.append(f"{status} **{name}** - {len(config.keywords)} keywords, threshold: {config.threshold}\n   Preview: {preview}")
        
        return "Configured topics:\n" + "\n".join(topic_list)
    
    elif query in topics:
        config = topics[query]
        return f"""Topic: **{query}**
Status: {'Enabled' if config.enabled else 'Disabled'}
Keywords: {', '.join(config.keywords)}
Threshold: {config.threshold}
Weight: {config.weight}
Reference text length: {len(config.reference_text)} characters
Reference text preview: {config.reference_text[:200]}..."""
    
    else:
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
    else:
        return f"Topic '{topic_name}' not found"

@tool
def test_topic_detection(test_text, cat):
    """Test topic detection on a given text. Input: text to analyze."""
    detected = detect_topics(test_text, cat)
    
    if not detected:
        return "No topics detected in the text"
    
    results = []
    for topic_name, topic_config, score in detected:
        results.append(f"- **{topic_name}**: score {score:.2f} (threshold: {topic_config.threshold})")
    
    return f"Detected topics:\n" + "\n".join(results)

@tool
def update_topic_keywords(update_data, cat):
    """Update keywords for an existing topic. Input: JSON with 'topic_name' and 'keywords' (list)."""
    try:
        data = json.loads(update_data)
        
        if not all(k in data for k in ["topic_name", "keywords"]):
            return "Error: Need both 'topic_name' and 'keywords' in JSON"
        
        topic_name = data["topic_name"]
        new_keywords = data["keywords"] if isinstance(data["keywords"], list) else [data["keywords"]]
        
        # Load current settings
        plugin = cat.mad_hatter.get_plugin()
        settings = plugin.load_settings()
        topics = json.loads(settings.get("topics_json", "{}"))
        
        if topic_name not in topics:
            return f"Error: Topic '{topic_name}' not found"
        
        # Update keywords
        topics[topic_name]["keywords"] = new_keywords
        
        # Save settings
        settings["topics_json"] = json.dumps(topics, ensure_ascii=False, indent=2)
        plugin.save_settings(settings)
        
        # Reload topics
        plugin_state["topics"] = load_topics_from_settings(settings)
        
        return f"✅ Updated keywords for topic '{topic_name}'. New keywords: {', '.join(new_keywords)}"
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    except Exception as e:
        return f"Error: {str(e)}"