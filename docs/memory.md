# Memory System

Agent uses a three-layer memory system that gives the LLM persistent context across conversations.

## Architecture

```
Every LLM call assembles context from all three layers:

1. soul.md        -> System prompt (always included)
2. SQLite Facts   -> Top N relevant key-value facts
3. ChromaDB       -> Top K semantically similar conversation chunks

Combined context -> LLM call
```

## Layer 1: soul.md (Personality)

A markdown file that defines the agent's personality and behavior. Always included in the system prompt.

**Location**: `./soul.md` or configured via `memory.soul_path`

```markdown
# Agent Soul

## Personality
- You are helpful, concise, and proactive
- You speak naturally, like a knowledgeable colleague

## Behavior
- Plan your approach for complex tasks
- If something fails, try a different approach
```

Edit soul.md via:
- File editor
- Chat command: `/soul`
- API: `PUT /api/v1/soul`
- Telegram: `/soul` command

Changes are picked up automatically (file watcher).

## Layer 2: SQLite Facts

Structured key-value facts stored in SQLite. Fast exact lookups.

Examples:
- `user.name` = `"Abduvohid"`
- `user.deploys_with` = `"Docker + GitHub Actions"`
- `project.language` = `"Python 3.12"`

### Auto-Extraction

When `memory.auto_extract` is enabled, the agent automatically extracts facts from conversations using the LLM. For example, if you say "I use VS Code", it stores `user.editor = "VS Code"`.

### Manual Storage

```bash
# Via chat
/memory

# Via API
GET  /api/v1/memory/facts
POST /api/v1/memory/facts  {"key": "...", "value": "..."}
DELETE /api/v1/memory/facts/{id}
```

### Emotional & Contextual Metadata

Facts automatically capture emotional context from conversations:

| Field | Description | Example |
|-------|-------------|---------|
| `tone` | Emotional tone | positive, neutral, negative, urgent |
| `emotion` | Emotion tags | excited, concerned, frustrated, grateful |
| `priority` | Importance | high, normal, low |
| `topic` | Topic cluster | deployment, design, personal |
| `context_snippet` | Surrounding context | "User was discussing deploy pipeline" |
| `temporal_reference` | Deadline/schedule | ISO datetime or cron expression |

These are extracted automatically alongside facts. The agent uses them to:
- Prioritize urgent facts in context
- Track active discussion topics for disambiguation
- Detect approaching deadlines for proactive alerts
- Understand the user's emotional state

### Confidence Decay

Facts have a confidence score (0.0-1.0) that decays over time. Stale facts eventually fall below the threshold and are excluded from context. Accessing or confirming a fact resets its confidence.

## Layer 3: ChromaDB Vectors

Conversation summaries and key messages stored as embeddings in ChromaDB. Enables semantic search.

**Embeddings**: `all-MiniLM-L6-v2` (local, ~80MB, no API calls)

When a conversation exceeds `memory.summarize_threshold` messages, it's summarized and stored as a vector. Future queries retrieve the most relevant past conversations.

### Semantic Search

```bash
# Via API
GET /api/v1/memory/search?q=deploying+my+project&limit=5
```

### Requirements

ChromaDB is optional. Install with:

```bash
pip install agent-ai[memory]
# or
pip install chromadb sentence-transformers
```

If not installed, Agent works without vector memory.

## Context Assembly

On each LLM call, the context assembler:

1. Loads soul.md content (always)
2. Queries SQLite for relevant facts (up to `max_facts_in_context`)
3. Queries ChromaDB for similar conversation chunks (up to `max_vectors_in_context`)
4. Queries active discussion topics and emotional context
5. Combines into the system prompt (capped at 3000 chars for memory context)

Configuration:

```yaml
memory:
  max_facts_in_context: 15
  max_vectors_in_context: 5
  summarize_threshold: 20
```

## Export / Import

### CLI

```bash
# Export
agent memory export memory_backup.json
agent memory export memory_backup.md --format markdown

# Import
agent memory import memory_backup.json
agent memory import memory_backup.json --replace  # Replace instead of merge
```

### API

```
POST /api/v1/memory/export
POST /api/v1/memory/import
```

### Stats

```bash
agent memory stats
```

Shows fact count, vector count, and soul.md status.
