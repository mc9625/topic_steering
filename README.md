# Topic Steering Plugin for Cheshire Cat AI 🎯

A sophisticated plugin that intelligently steers conversations by injecting high-priority context when topics are detected. Uses flexible semantic understanding to interpret user intent naturally, without rigid keyword matching.

## Version 2.0 Features 🌟

- **🧠 Flexible Semantic Understanding**: Interprets user intent, not just keywords
- **📚 Context Injection System**: Adds reference texts as high-priority context for the LLM
- **📁 File Loading Support**: Load reference texts directly from files
- **⚡ Smart Priority System**: Instructs LLM to prioritize injected context over memories
- **🎛️ Highly Configurable**: Fine-tune detection sensitivity and context priority
- **🔍 Multiple Detection Modes**: Semantic AI, keyword, or hybrid approaches

## How It Works 🔄

### 1. **Semantic Topic Detection**
The plugin doesn't just look for exact keywords. It understands the **intent and meaning** behind user messages using AI embeddings. For example:
- User asks about "military strategy" → Detects "war" topic
- User mentions "battle tactics" → Detects "war" topic
- User discusses "peaceful resolution" → May not trigger "war" topic

### 2. **Context Injection (Not Response Replacement)**
Instead of replacing the Cat's response, the plugin:
1. Detects relevant topics in the user's message
2. Injects reference texts as **high-priority context** before the LLM processes
3. Instructs the LLM to give this context **more weight** than episodic/declarative memories
4. The LLM naturally incorporates this context into its response

### 3. **File-Based Content Management**
You can now load reference texts from files, making it easy to manage large amounts of content without editing JSON.

## Installation 📦

1. Download the plugin files
2. Place them in your Cheshire Cat's `plugins/topic_steering` folder
3. Restart the Cat or reload plugins
4. Configure your topics in the plugin settings

## Configuration 🔧

### Plugin Settings

- **`topics_json`**: JSON configuration of all topics
- **`enabled`**: Master on/off switch
- **`detection_mode`**: 
  - `semantic`: AI-powered understanding (recommended)
  - `keyword`: Simple keyword matching
  - `hybrid`: Best of both worlds
- **`context_priority`**: How strongly to weight injected context (0.0-1.0)
  - 0.9 = Very high priority (default)
  - 0.5 = Equal to memories
  - 0.1 = Light suggestion
- **`max_topics_per_message`**: Topics to process per message (1-5)
- **`debug_mode`**: Enable detailed logging

### Topic Configuration Format

```json
{
  "topic_name": {
    "keywords": ["keyword1", "keyword2", "related phrase"],
    "threshold": 0.65,
    "reference_text": "The authoritative content on this topic...",
    "weight": 0.8,
    "enabled": true
  }
}
```

- **`keywords`**: Words/phrases that help identify the topic (used for semantic understanding)
- **`threshold`**: Similarity threshold (0-1). Lower = more sensitive
- **`reference_text`**: The content to inject as context
- **`weight`**: Individual topic weight (kept for compatibility)
- **`enabled`**: Toggle topics on/off

## Tools & Commands 🛠️

### Core Topic Management

**`add_topic`** - Create or update a topic:
```
Input: {"name": "philosophy", "keywords": ["philosophy", "ethics", "morality", "existentialism"], "reference_text": "Philosophy is the study of fundamental questions..."}
```

**`list_topics`** - View all topics or get details:
```
Input: "all"  # Lists all topics with previews
Input: "war"  # Shows full details for "war" topic
```

**`remove_topic`** - Delete a topic:
```
Input: "philosophy"
```

### File Management

**`load_topic_from_file`** - Load reference text from a file:
```
Input: {"topic_name": "history", "file_path": "/path/to/history_context.txt"}
```

### Testing & Maintenance

**`test_topic_detection`** - Test detection on sample text:
```
Input: "Tell me about the horrors of war and military conflicts"
```

**`update_topic_keywords`** - Update keywords for better detection:
```
Input: {"topic_name": "war", "keywords": ["war", "battle", "conflict", "military", "soldier", "army"]}
```

## Usage Examples 💡

### Example 1: Historical Context (Benvenuto Cellini on War)

The default configuration includes your war narrative. When users ask about war, battles, or military topics, the Cat will:

1. **Detect** the war topic through semantic understanding
2. **Inject** Cellini's first-person account as high-priority context
3. **Generate** a response that naturally incorporates this historical perspective

User: "What do you think about war?"
→ The Cat responds incorporating Cellini's personal experience from the 1527 Siege of Rome

### Example 2: Company Knowledge Base

```json
{
  "company_policies": {
    "keywords": ["policy", "procedure", "company rules", "guidelines", "HR"],
    "threshold": 0.6,
    "reference_text": "[Load from file with company policies]",
    "weight": 0.95,
    "enabled": true
  }
}
```

Load the full policy document:
```
load_topic_from_file {"topic_name": "company_policies", "file_path": "/docs/company_policies.txt"}
```

### Example 3: Educational Content

```json
{
  "quantum_physics": {
    "keywords": ["quantum", "physics", "particle", "wave function", "superposition"],
    "threshold": 0.7,
    "reference_text": "[Authoritative quantum physics explanation]",
    "weight": 0.85,
    "enabled": true
  }
}
```

## Best Practices 💡

### 1. **Keyword Selection for Semantic Understanding**
- Include core terms AND related concepts
- Add synonyms in multiple languages if needed
- Think about how users might ask about the topic

### 2. **Threshold Tuning**
- Start with 0.65-0.70 for general topics
- Use 0.75-0.85 for specific technical topics
- Lower to 0.55-0.65 for broad concepts

### 3. **Context Priority**
- 0.9-1.0: Critical information (policies, facts)
- 0.7-0.8: Important context (historical accounts)
- 0.5-0.6: Supplementary information

### 4. **Reference Text Guidelines**
- Write in a neutral, informative tone
- Include key facts and perspectives
- Keep it focused on the topic
- Can be first-person narratives (like Cellini example)

## Advanced Features 🚀

### Semantic Understanding
The plugin uses AI embeddings to understand meaning, not just match keywords:
- "military strategy" → relates to "war"
- "ancient Rome" → might trigger historical topics
- "conflict resolution" → understood in context

### Context Weighting
The LLM is explicitly instructed to prioritize injected context:
```
"Give this context MORE weight than episodic or declarative memories"
```

### Multi-User Support
Each user's detected topics are tracked separately, allowing personalized experiences in multi-user environments.

## Troubleshooting 🔍

**Topics not detected?**
- Check if semantic mode is enabled
- Lower the threshold
- Add more diverse keywords
- Test with `test_topic_detection`

**Context not appearing in responses?**
- Verify context_priority is high (>0.8)
- Check debug logs
- Ensure topic is enabled

**File loading issues?**
- Check file path is correct
- Ensure file encoding is UTF-8
- Verify read permissions

## Technical Details 🔧

### Hook Usage (see hooks.md)
- `before_cat_reads_message`: Detects topics early
- `before_agent_starts`: Injects context into agent input
- `agent_prompt_instructions`: Modifies LLM instructions for priority
- `after_cat_recalls_memories`: Cleanup

### Why This Approach?
- **Semantic detection**: More natural and flexible than keywords
- **Context injection**: Preserves Cat's personality while adding knowledge
- **Priority system**: Ensures important information is emphasized

## Version History 📝

**v2.0.0** (Current)
- Semantic understanding for flexible topic detection
- Context injection system replacing response modification
- File loading support
- Priority weighting for injected context
- Per-user topic tracking

**v1.0.0**
- Initial release with basic keyword detection
- Response modification approach

## Contributing 🤝

Feel free to submit issues, feature requests, or pull requests to improve this plugin!

## License 📄

This plugin is released under the same license as the Cheshire Cat AI project.

---

Made with ❤️ for the Cheshire Cat AI community by Claude the Cat