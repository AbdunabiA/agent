# Multi-Agent Orchestration

Agent includes a multi-agent orchestration system that lets you delegate complex work to teams of specialized sub-agents. Each sub-agent has its own persona, tool permissions, and iteration budget.

## Quick Start

Enable orchestration in `agent.yaml`:

```yaml
orchestration:
  enabled: true
  teams_directory: "teams"
  max_concurrent_agents: 5
```

Then ask the agent:

```
Run the full_feature project to build a user authentication system
```

The agent will automatically spawn the right sub-agents from your team definitions and coordinate their work through the pipeline.

## Concepts

### Teams

A **team** is a YAML file defining a group of specialized roles. Teams live in the `teams/` directory and are auto-discovered at startup.

```
teams/
  engineering.yaml    # architect, backend_developer, frontend_developer, ...
  product.yaml        # product_manager, ux_designer, business_analyst, ...
  quality.yaml        # qa_engineer, security_reviewer, integration_tester, ...
  content.yaml        # technical_writer, copywriter, api_docs_writer, ...
  data.yaml           # data_engineer, ml_engineer, data_analyst
  design.yaml         # ui_designer, accessibility_specialist
  research.yaml       # tech_researcher, reflexion_agent
```

### Roles

A **role** is a single sub-agent definition within a team. Each role has:

- **Persona**: Detailed system prompt defining identity, deliverables, success metrics, anti-patterns, and workflow
- **Tool permissions**: Which tools the role can and cannot use
- **Iteration budget**: Maximum number of tool-call loops

### Projects

A **project** is a multi-stage pipeline that orchestrates roles across teams. Projects live in `teams/projects/` and define sequential and parallel stages with optional feedback loops.

### Controller

The optional **controller** is a project-manager agent that receives high-level work orders and autonomously decomposes, delegates, and monitors sub-agent work.

## Team YAML Format

```yaml
name: engineering
description: "Engineering team — architecture, backend, frontend, DevOps"
roles:
  - name: backend_developer
    persona: |
      ## Identity
      You are the backend engine — you turn architecture into running async Python.
      You live and breathe asyncio, type hints, and Pydantic.

      ## Deliverables
      - Working async Python modules following existing conventions
      - Unit tests for every new function

      ## Success Metrics
      1. All new code passes the linter
      2. Tests cover happy path and error cases

      ## Anti-patterns
      - Never use print() — use structlog
      - Never catch bare Exception

      ## Workflow
      1. Read existing code to understand context
      2. Implement the feature
      3. Write tests
      4. Run linter and tests
    allowed_tools:
      - file_read
      - file_write
      - shell_exec
      - python_exec
      - get_my_tasks
      - complete_my_task
      - assign_task
      - ask_team
      - request_review
    max_iterations: 200

  - name: code_reviewer
    persona: |
      You are the standards guardian...
    allowed_tools: [file_read, shell_exec, web_search]
    denied_tools: [file_write]  # Read-only role
    max_iterations: 200
```

### Role Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique role identifier within the team |
| `persona` | string | Yes | System prompt defining the role's behavior |
| `allowed_tools` | list[str] | No | Tools available to this role (empty = all safe tools) |
| `denied_tools` | list[str] | No | Tools explicitly denied (overrides allowed) |
| `model` | string | No | Override LLM model for this role |
| `max_iterations` | int | No | Max tool-call loops (default: from orchestration config) |

### Persona Best Practices

A good persona includes these sections:

1. **Identity**: Who the agent is, their expertise, pet peeves
2. **Deliverables**: Concrete outputs expected
3. **Success Metrics**: Measurable criteria for quality
4. **Anti-patterns**: What NOT to do
5. **Workflow**: Step-by-step process to follow
6. **Collaboration Protocol**: How to interact with other roles
7. **Handoff Rules**: When to delegate to whom

## Project Pipeline Format

Projects define multi-stage workflows where each stage runs one or more agents:

```yaml
name: full_feature
description: "End-to-end feature development"
stages:
  - name: requirements
    parallel: true
    agents:
      - team: product
        role: product_manager
      - team: product
        role: business_analyst

  - name: design
    parallel: true
    agents:
      - team: engineering
        role: architect
      - team: product
        role: ux_designer

  - name: implementation
    parallel: true
    agents:
      - team: engineering
        role: backend_developer
      - team: engineering
        role: frontend_developer

  - name: testing
    parallel: true
    agents:
      - team: quality
        role: qa_engineer
      - team: quality
        role: integration_tester
    denied_tools: [assign_task, request_review]
    feedback:
      fix_stage: fix_issues
      max_retries: 3

  - name: fix_issues
    feedback_target: true
    parallel: true
    agents:
      - team: engineering
        role: backend_developer
      - team: engineering
        role: frontend_developer

  - name: documentation
    parallel: true
    agents:
      - team: content
        role: technical_writer
      - team: content
        role: api_docs_writer
```

### Stage Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | Required | Stage identifier |
| `parallel` | bool | `false` | Run agents in parallel or sequentially |
| `agents` | list | Required | Agents to run (team + role references) |
| `denied_tools` | list[str] | `[]` | Tools denied for all agents in this stage |
| `feedback` | object | `null` | Feedback loop config |
| `feedback_target` | bool | `false` | Mark stage as fix target (only entered via feedback) |
| `mode` | string | `null` | Set to `"discussion"` for consensus mode |
| `discussion` | object | `null` | Discussion config (rounds, moderator, consensus) |

### Feedback Loops

Feedback loops create automated test-fix cycles:

```yaml
- name: testing
  agents:
    - team: quality
      role: qa_engineer
  feedback:
    fix_stage: fix_issues  # Jump to this stage on failure
    max_retries: 3         # Max fix attempts before giving up

- name: fix_issues
  feedback_target: true    # Only entered via feedback loop
  agents:
    - team: engineering
      role: backend_developer
```

**Flow**: testing runs → if tests fail → jump to fix_issues → re-run testing → repeat up to `max_retries`

### Discussion Mode

Discussion mode enables multi-agent consensus:

```yaml
- name: architecture_review
  mode: discussion
  agents:
    - team: engineering
      role: architect
    - team: engineering
      role: api_designer
  discussion:
    rounds: 3
    moderator:
      team: engineering
      role: architect
    consensus_required: true
```

**Flow**: All agents respond in parallel → moderator summarizes → agents see transcript and respond again → repeat for N rounds or until consensus.

### Context Passing Between Stages

Each stage's combined output automatically becomes the next stage's input context. This means:

- **requirements** stage output → feeds into **design** stage as context
- **design** stage output → feeds into **implementation** stage as context
- Context is bounded to ~50KB to prevent token explosion

## Available Project Pipelines

| Pipeline | Stages | Description |
|----------|--------|-------------|
| `quick_task` | 1-2 | Simple single-agent task |
| `bug_fix` | 3 | Investigate → fix → test |
| `code_review` | 2 | Review → report |
| `full_feature` | 8 | Requirements → design → build → test → fix → security → docs |
| `refactor` | 4 | Analyze → plan → implement → verify |
| `build_app` | 6 | Design → build → test → deploy |
| `build_saas` | 12 | Full SaaS product pipeline |
| `build_platform` | 10+ | Platform development |
| `mobile_app` | 6 | Mobile app development |
| `data_pipeline` | 5 | Data engineering pipeline |
| `launch_prep` | 4 | Pre-launch checklist |

## Controller Agent

The controller is an optional project-manager agent that autonomously handles complex work orders:

```yaml
orchestration:
  enabled: true
  use_controller: true
  controller_model: "claude-sonnet-4-5-20250929"  # Optional model override
  controller_max_turns: 200
```

When enabled, the controller:

1. **Receives** a work order from the main agent
2. **Analyzes** the request and decides the approach (single agent, parallel agents, team, or pipeline)
3. **Spawns** appropriate sub-agents with detailed instructions
4. **Monitors** progress and handles failures
5. **Reports** results back to the main agent

The controller never does substantive work itself — it only coordinates and delegates.

## Inter-Agent Communication

Sub-agents can communicate through several mechanisms:

### Task Board

Agents post and claim tasks through a shared board:

```
report_bug(title, description, severity)     # Report a bug to be fixed
request_review(description, assignee)         # Request code review
assign_task(title, description, assignee)     # Assign work to another agent
```

### Direct Consultation

At nesting depth 0, agents can consult each other:

```
consult_agent(team, role, question, context)       # Ask for expert opinion
delegate_to_specialist(team, role, instruction)     # Delegate specific work
```

### Event Bus

All agent activity emits events for observability:

- `SUBAGENT_SPAWNED`, `SUBAGENT_COMPLETED`, `SUBAGENT_FAILED`
- `PROJECT_STARTED`, `PROJECT_STAGE_COMPLETED`, `PROJECT_COMPLETED`
- `DISCUSSION_STARTED`, `DISCUSSION_CONSENSUS_REACHED`

## Configuration Reference

```yaml
orchestration:
  enabled: false              # Enable multi-agent orchestration
  max_concurrent_agents: 5    # Max parallel sub-agents
  default_max_iterations: 200 # Default iteration budget per agent
  subagent_timeout: 1800      # Timeout per agent in seconds (30 min)
  teams_directory: "teams"    # Directory containing team YAML files
  use_controller: false       # Enable the controller agent
  controller_model: null      # Model override for controller
  controller_max_turns: 200   # Max turns for controller
  teams: []                   # Inline team definitions (teams/ dir takes precedence)
```

## Safety & Limits

- **Tool isolation**: Sub-agents get scoped tool registries. Orchestration tools (spawn_subagent, run_project) are always excluded to prevent infinite recursion.
- **Nesting depth**: Agents at depth >= 1 cannot consult or delegate to other agents.
- **Hard depth limit**: Maximum nesting depth of 3 prevents runaway spawning.
- **Concurrency cap**: `max_concurrent_agents` enforced via async locks.
- **Timeout**: Each sub-agent has a configurable timeout (default 30 minutes).
- **Result pruning**: Completed results are pruned to prevent unbounded memory growth.

## Creating Custom Teams

1. Create a YAML file in `teams/`:

```yaml
# teams/my_team.yaml
name: my_team
description: "My custom team for specific project needs"
roles:
  - name: specialist
    persona: |
      You are a specialist in [domain]...
    allowed_tools: [file_read, file_write, shell_exec]
    max_iterations: 100
```

2. Restart the agent (teams are loaded at startup).

3. Use your team:

```
Spawn a specialist from my_team to analyze the codebase
```

Or reference it in a project pipeline:

```yaml
# teams/projects/my_pipeline.yaml
name: my_pipeline
stages:
  - name: analyze
    agents:
      - team: my_team
        role: specialist
```
