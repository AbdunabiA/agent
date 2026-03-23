# Database Migrations

Agent uses SQLite for persistent storage. The schema evolves over time and is automatically migrated on startup.

## How It Works

On every startup, Agent checks the current schema version in the database and applies any pending migrations. No manual intervention is required.

**Always back up your database before upgrading Agent:**

```bash
cp data/agent.db data/agent.db.backup
```

## Checking the Current Version

```bash
sqlite3 data/agent.db "SELECT version FROM schema_version"
```

## Schema Version History

| Version | Changes |
|---------|---------|
| **v1** | Initial schema — `facts`, `conversations`, `messages`, `audit_log` tables |
| **v2** | Added `scheduled_tasks` table for cron jobs and reminders |
| **v3** | Added indexes on `audit_log` for faster queries |
| **v4** | Added `working_memory` and `tracer` tables |
| **v5** | Added `task_board` table for persistent task tracking |
| **v6** | Added `cost_entries` table for LLM cost tracking |

## Database Location

The database file is stored at `data/agent.db` relative to the project root. This directory is gitignored.

## Manual Recovery

If a migration fails (e.g., due to a corrupted database):

1. Stop the agent:

```bash
agent stop
```

2. Restore from backup:

```bash
cp data/agent.db.backup data/agent.db
```

3. Restart:

```bash
agent start
```

If no backup is available, you can reset the database entirely. This will lose all stored data:

```bash
rm data/agent.db
agent start
```

The agent will create a fresh database with the latest schema on startup.
