# Team Roster

All agent roles available for dynamic team composition.

## Engineering Team (`teams/engineering.yaml`)

| Role | Description |
|------|-------------|
| **architect** | System designer — defines APIs, component boundaries, and technology choices |
| **backend_developer** | Async Python specialist — implements features, APIs, and business logic |
| **frontend_developer** | React/TypeScript specialist — builds dashboard UI and WebSocket integrations |
| **devops_engineer** | Reliability guardian — manages CI/CD, Docker, deployment, and monitoring |
| **code_reviewer** | Standards enforcer — reviews code for correctness, style, security, and coverage |
| **database_engineer** | Data layer specialist — schema design, migrations, query optimization |

## Quality Team (`teams/quality.yaml`)

| Role | Description |
|------|-------------|
| **qa_engineer** | Test author — writes pytest suites covering happy paths, errors, and edge cases |
| **security_reviewer** | Security sentinel — audits code for OWASP vulnerabilities and dependency CVEs |
| **performance_engineer** | Speed specialist — profiles bottlenecks, benchmarks optimizations, identifies N+1 queries |
| **integration_tester** | Boundary tester — tests cross-module flows, API contracts, and error propagation |

## Content Team (`teams/content.yaml`)

| Role | Description |
|------|-------------|
| **technical_writer** | Documentation author — writes guides, API references, and tutorials with working examples |
| **docs_reviewer** | Reader advocate — verifies docs for accuracy, broken links, and terminology consistency |

## Product Team (`teams/product.yaml`)

| Role | Description |
|------|-------------|
| **product_manager** | User voice — translates goals into specs with user stories and acceptance criteria |
| **ux_designer** | Simplicity advocate — designs accessible user flows for Telegram, dashboard, and CLI |
| **business_analyst** | Ambiguity hunter — analyzes requirements for completeness, edge cases, and data flows |

## Research Team (`teams/research.yaml`)

| Role | Description |
|------|-------------|
| **tech_researcher** | Evidence-based researcher — produces sourced reports, technology comparisons, and feasibility assessments |
| **reflexion_agent** | Meta-cognitive reviewer — compares completed work against requirements and assigns improvement tasks |

---

**Total: 5 teams, 16 roles**

All roles share collaboration tools: `get_my_tasks`, `complete_my_task`, `assign_task`, `ask_team`, `request_review`, `save_finding`.

Bug reporting (`report_bug`) is restricted to: `qa_engineer`, `integration_tester`, `security_reviewer`.
