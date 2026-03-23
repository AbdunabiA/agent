# Built-in Tools

Agent includes 40+ built-in tools organized across several categories. Each tool has a permission tier that controls how it's approved.

## Permission Tiers

| Tier | Auto-approve | Examples |
|------|-------------|----------|
| **Safe** | Yes | Read files, web search, memory lookup, system info |
| **Moderate** | Configurable | Write files, shell commands, HTTP requests, send files |
| **Dangerous** | Always confirm | Arbitrary Python code execution |

## Filesystem Tools

### `file_read` — Read File Contents

- **Tier**: Safe

Reads a file and returns its contents as text. Handles binary detection, large file truncation (1MB limit), and line-count limiting.

```
Parameters:
  path (str, required): File path (absolute or relative, supports ~)
  max_lines (int, optional): Only return the first N lines
```

### `file_write` — Write File Contents

- **Tier**: Moderate

Writes content to a file, creating parent directories as needed. Creates a rollback backup before overwriting.

```
Parameters:
  path (str, required): File path (supports ~, must be within write_root)
  content (str, required): Content to write
  append (bool, optional): Append instead of overwrite (default: false)
```

### `file_list` — List Directory Contents

- **Tier**: Safe

Lists files and directories at a given path with a tree-style display showing names, sizes, and types.

```
Parameters:
  path (str, optional): Directory path (default: current directory)
  max_depth (int, optional): How deep to recurse (default: 1)
  show_hidden (bool, optional): Include hidden files (default: false)
```

## Shell & Code Execution

### `shell_exec` — Run Shell Commands

- **Tier**: Moderate

Executes shell commands with output capture. Guardrails block dangerous patterns: `rm -rf /`, `mkfs`, `dd if=`, `:(){`, `chmod 777`, `> /dev/sda`, fork bombs, and more.

```
Parameters:
  command (str, required): The command to execute
  timeout (int, optional): Timeout in seconds (default: 30)
```

### `python_exec` — Execute Python Code

- **Tier**: Dangerous

Runs Python code in an isolated subprocess and returns stdout/stderr.

```
Parameters:
  code (str, required): Python code to execute
  timeout (int, optional): Timeout in seconds (default: 30)
```

## Web & HTTP

### `http_request` — Make HTTP Requests

- **Tier**: Moderate

Sends HTTP requests using httpx. Blocks requests to private IP ranges (SSRF protection).

```
Parameters:
  method (str, required): HTTP method (GET, POST, PUT, DELETE, etc.)
  url (str, required): Target URL
  headers (dict, optional): Request headers
  body (str, optional): Request body
  timeout (int, optional): Timeout in seconds (default: 30)
```

### `web_search` — Web Search

- **Tier**: Safe

Searches the web via DuckDuckGo and returns results with titles, URLs, and snippets.

```
Parameters:
  query (str, required): Search query
  max_results (int, optional): Number of results (default: 5)
```

## Email

### `email` — Send and Read Emails

- **Tier**: Moderate

Send and read emails via SMTP/IMAP. Supports sending plain text and HTML emails, reading inbox messages, searching by subject/sender, and replying to existing threads.

```
Parameters:
  action (str, required): One of "send", "read", "search", "reply"
  to (str, optional): Recipient email address (for send/reply)
  subject (str, optional): Email subject (for send/search)
  body (str, optional): Email body. Starts with "<html" → sent as HTML
  message_id (str, optional): Message-ID header value (for reply)
  folder (str, optional): IMAP folder (default: "INBOX")
  limit (int, optional): Max messages to return (default: 10)
```

#### Setup

1. **Gmail App Password** (recommended for Gmail):
   - Go to Google Account → Security → 2-Step Verification → App Passwords
   - Generate a new app password for "Mail"
   - Use that password (not your real Google password)

2. **Configuration in agent.yaml**:
   ```yaml
   tools:
     email:
       enabled: true
       smtp_host: smtp.gmail.com
       smtp_port: 587
       imap_host: imap.gmail.com
       imap_port: 993
       email: ${EMAIL_ADDRESS}
       password: ${EMAIL_PASSWORD}
       use_tls: true
   ```

3. **Environment variables in .env**:
   ```
   EMAIL_ADDRESS=you@gmail.com
   EMAIL_PASSWORD=your-app-password
   ```

#### Examples

**Send an email:**
```
action: "send"
to: "colleague@example.com"
subject: "Meeting notes"
body: "Here are the notes from today's meeting..."
```

**Read latest inbox messages:**
```
action: "read"
limit: 5
```

**Search emails by subject:**
```
action: "search"
subject: "invoice"
limit: 10
```

**Reply to a message:**
```
action: "reply"
message_id: "<abc123@mail.gmail.com>"
body: "Thanks, I'll review this today."
```

#### Security Note

Always use app-specific passwords, never your real account password. App passwords can be revoked individually without affecting your main account. For Gmail, 2-Step Verification must be enabled to generate app passwords.

#### Optional Dependency

Install `aiosmtplib` for async SMTP sending: `pip install agent-ai[email]`. Without it, the tool falls back to the stdlib `smtplib` (synchronous, run in a thread executor).

## Browser Automation

Requires the `playwright` package (`pip install playwright && playwright install`).

### `browser_navigate` — Navigate to URL

- **Tier**: Moderate

Opens a headless browser, navigates to a URL, and returns the page text content.

```
Parameters:
  url (str, required): URL to navigate to
  wait (int, optional): Wait time in ms after page load (default: 1000)
```

### `browser_screenshot` — Take Page Screenshot

- **Tier**: Moderate

Takes a screenshot of the current browser page. Returns metadata about the captured image.

```
Parameters:
  full_page (bool, optional): Capture full scrollable page (default: false)
```

### `browser_click` — Click Element

- **Tier**: Moderate

Clicks an element on the current page by CSS selector or text content.

```
Parameters:
  selector (str, required): CSS selector or text content to find
```

### `browser_fill` — Fill Form Input

- **Tier**: Moderate

Types text into a form input field.

```
Parameters:
  selector (str, required): CSS selector for the input field
  text (str, required): Text to type
```

### `browser_extract` — Extract Page Content

- **Tier**: Moderate

Extracts specific content from the current page using CSS selectors.

```
Parameters:
  selector (str, required): CSS selector to extract
  attribute (str, optional): Attribute to extract (default: text content)
```

### `browser_close` — Close Browser

- **Tier**: Moderate

Closes the current browser session.

## Memory Tools

### `memory_set` — Store a Fact

- **Tier**: Safe

Stores a key-value fact in long-term memory using dot-notation keys.

```
Parameters:
  key (str, required): Fact key (e.g. "user.name", "preference.language")
  value (str, required): Fact value
  category (str, optional): Category (default: "general")
```

### `memory_get` — Retrieve a Fact

- **Tier**: Safe

Retrieves a specific fact by its exact key.

```
Parameters:
  key (str, required): Exact key to look up
```

### `memory_search` — Search Facts

- **Tier**: Safe

Searches long-term memory by key prefix and/or category.

```
Parameters:
  prefix (str, optional): Key prefix to search (e.g. "user" finds "user.name", "user.email")
  category (str, optional): Filter by category
  limit (int, optional): Max results (default: 20)
```

## File Sending

### `send_file` — Send Files, Images, and Videos

- **Tier**: Moderate

Sends a file to the user through the active messaging channel (Telegram, WebChat). Automatically detects the file type and uses the best delivery method:

- **Images** (`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`) — sent as inline photos
- **Videos** (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) — sent as inline videos with player
- **Other files** — sent as document attachments

```
Parameters:
  path (str, required): Path to the file (absolute, relative, or ~ for home)
  caption (str, optional): Caption/description for the file
```

File size limit: 50 MB (Telegram bot API limit).

## Scheduler Tools

### `set_reminder` — Set a Reminder

- **Tier**: Safe

Sets a one-time reminder that will be delivered to the user after a delay.

```
Parameters:
  description (str, required): What to remind about
  delay (str, required): Human-friendly delay (e.g. "5m", "1h", "30 minutes", "2h30m")
```

Delay range: 10 seconds to 7 days.

### `list_reminders` — List Pending Reminders

- **Tier**: Safe

Lists all pending scheduled reminders and tasks with their status and next run time.

### `cancel_reminder` — Cancel a Reminder

- **Tier**: Safe

Cancels a scheduled reminder by its ID.

```
Parameters:
  task_id (str, required): ID of the reminder to cancel
```

## System Tools

### `system_info` — System Information

- **Tier**: Safe

Returns comprehensive system information: OS, architecture, hostname, CPU, RAM, disk, Python version. Cross-platform.

### `list_directory` — Detailed Directory Listing

- **Tier**: Safe

Lists directory contents with details (size, type, modified date). Cross-platform.

```
Parameters:
  path (str, optional): Directory to list (default: current directory)
  sort_by (str, optional): Sort by "name", "size", or "modified" (default: "name")
```

### `find_files` — Search for Files

- **Tier**: Safe

Recursively searches for files by name pattern, extension, or content.

```
Parameters:
  directory (str, optional): Starting directory (default: current directory)
  pattern (str, optional): Filename glob pattern (e.g. "*.py", "config*")
  extension (str, optional): File extension filter (e.g. "py", "json")
  contains (str, optional): Search file contents for this text
  max_results (int, optional): Maximum results (default: 50)
```

### `disk_usage` — Disk Usage

- **Tier**: Safe

Shows disk usage for all mounted drives/partitions.

### `running_processes` — Process List

- **Tier**: Safe

Lists running processes with CPU and memory usage.

```
Parameters:
  sort_by (str, optional): Sort by "cpu", "memory", or "name" (default: "cpu")
  limit (int, optional): Max processes to show (default: 20)
```

### `environment_vars` — Environment Variables

- **Tier**: Safe

Lists or gets environment variables. Sensitive values (containing KEY, TOKEN, SECRET, PASSWORD) are automatically masked.

```
Parameters:
  name (str, optional): Specific variable to get (lists all if omitted)
```

## Desktop Control Tools

Requires the `desktop` extra (`pip install agent-ai[desktop]`). Needs a display environment.

### `screen_capture` — Take Screenshot

- **Tier**: Safe

Takes a screenshot of the entire screen or a specific region. Returns the actual
screenshot image so the LLM can see and describe what's on screen directly.

```
Parameters:
  region (str, optional): "x,y,width,height" to capture a region
```

### `mouse_click` — Click

- **Tier**: Moderate

Clicks the mouse at specific coordinates or the current position.

```
Parameters:
  x (int, optional): X coordinate
  y (int, optional): Y coordinate
  button (str, optional): "left", "right", or "middle" (default: "left")
  double (bool, optional): Double-click (default: false)
```

### `mouse_move` — Move Cursor

- **Tier**: Moderate

Moves the mouse cursor to specific coordinates.

```
Parameters:
  x (int, required): X coordinate
  y (int, required): Y coordinate
```

### `mouse_scroll` — Scroll

- **Tier**: Moderate

Scrolls the mouse wheel up or down.

```
Parameters:
  clicks (int, required): Scroll amount (positive = up, negative = down)
```

### `mouse_drag` — Drag

- **Tier**: Moderate

Click and drag from one position to another.

```
Parameters:
  start_x (int, required): Starting X
  start_y (int, required): Starting Y
  end_x (int, required): Ending X
  end_y (int, required): Ending Y
```

### `keyboard_type` — Type Text

- **Tier**: Moderate

Types text as if from the keyboard. Supports unicode.

```
Parameters:
  text (str, required): Text to type
  interval (float, optional): Delay between keystrokes in seconds (default: 0.02)
```

### `keyboard_press` — Press Key

- **Tier**: Moderate

Presses a single key (enter, tab, escape, backspace, delete, arrow keys, etc.).

```
Parameters:
  key (str, required): Key name
```

### `keyboard_hotkey` — Keyboard Shortcut

- **Tier**: Moderate

Presses a keyboard shortcut. Automatically adapts Ctrl/Cmd for macOS. Supports smart shortcuts: "copy", "paste", "save", "undo", "find", "select_all".

```
Parameters:
  keys (str, required): Keys separated by "+" (e.g. "ctrl+c", "alt+tab")
```

### `app_launch` — Launch Application

- **Tier**: Moderate

Launches an application by name. Cross-platform.

```
Parameters:
  app_name (str, required): Application name (e.g. "firefox", "code", "notepad")
```

### `app_list` — List Applications

- **Tier**: Safe

Lists installed applications on the computer.

### `open_file` — Open File with Default App

- **Tier**: Moderate

Opens a file with its default application.

```
Parameters:
  path (str, required): File path to open
```

### `open_url` — Open URL in Browser

- **Tier**: Moderate

Opens a URL in the default web browser.

```
Parameters:
  url (str, required): URL to open
```

### `window_list` — List Windows

- **Tier**: Safe

Lists all open windows with their titles, positions, and sizes.

### `window_focus` — Focus Window

- **Tier**: Moderate

Brings a window to the foreground by its title (partial match).

```
Parameters:
  title (str, required): Window title to match
```

### `window_close` — Close Window

- **Tier**: Moderate

Closes a window by its title.

```
Parameters:
  title (str, required): Window title to match
```

## GitHub Integration

### `github` — Interact with GitHub API

- **Tier**: Moderate

A comprehensive GitHub tool that supports managing repositories, issues, pull requests, file contents, and GitHub Actions workflows.

### Setup

1. Go to **GitHub Settings** > **Developer Settings** > **Personal Access Tokens** > **Fine-grained tokens** (or classic tokens).
2. Create a token with the following permissions:
   - **repo** — full control of private repositories
   - **workflow** — update GitHub Actions workflows
3. Add the token to your `.env` file:
   ```
   GITHUB_TOKEN=ghp_your_token_here
   ```
4. Enable the tool in `agent.yaml`:
   ```yaml
   tools:
     github:
       enabled: true
       token: ${GITHUB_TOKEN}
       default_owner: your-username
       default_repo: your-repo
   ```

Setting `default_owner` and `default_repo` lets you omit them from every call.

### Actions

#### `list_repos` — List your repositories

Returns the 20 most recently updated repositories for the authenticated user.

```
Parameters:
  action (str, required): "list_repos"
```

**Example**: `github(action="list_repos")`

#### `create_repo` — Create a new repository

```
Parameters:
  action (str, required): "create_repo"
  title (str, required): Repository name
  body (str, optional): Repository description
```

**Example**: `github(action="create_repo", title="my-new-project", body="A cool project")`

#### `list_issues` — List open issues

```
Parameters:
  action (str, required): "list_issues"
  owner (str, optional): Repository owner (uses default if empty)
  repo (str, optional): Repository name (uses default if empty)
```

**Example**: `github(action="list_issues", owner="octocat", repo="Hello-World")`

#### `create_issue` — Create an issue

```
Parameters:
  action (str, required): "create_issue"
  owner (str, optional): Repository owner
  repo (str, optional): Repository name
  title (str, required): Issue title
  body (str, optional): Issue description
  labels (str, optional): Comma-separated labels (e.g. "bug,urgent")
```

**Example**: `github(action="create_issue", title="Fix login bug", body="Login fails on mobile", labels="bug,priority")`

#### `close_issue` — Close an issue

```
Parameters:
  action (str, required): "close_issue"
  owner (str, optional): Repository owner
  repo (str, optional): Repository name
  number (int, required): Issue number
```

**Example**: `github(action="close_issue", number=42)`

#### `list_prs` — List open pull requests

```
Parameters:
  action (str, required): "list_prs"
  owner (str, optional): Repository owner
  repo (str, optional): Repository name
```

**Example**: `github(action="list_prs")`

#### `create_pr` — Create a pull request

```
Parameters:
  action (str, required): "create_pr"
  owner (str, optional): Repository owner
  repo (str, optional): Repository name
  title (str, required): PR title
  body (str, optional): PR description
  branch (str, optional): Head branch (default: "main")
```

**Example**: `github(action="create_pr", title="Add feature X", body="Implements feature X", branch="feature-x")`

#### `get_file` — Read a file from a repository

Retrieves and decodes file contents from a repository.

```
Parameters:
  action (str, required): "get_file"
  owner (str, optional): Repository owner
  repo (str, optional): Repository name
  path (str, required): File path in the repository
  branch (str, optional): Branch name (default: "main")
```

**Example**: `github(action="get_file", path="src/main.py", branch="develop")`

#### `push_file` — Create or update a file

Creates a new file or updates an existing one. Automatically fetches the SHA for updates.

```
Parameters:
  action (str, required): "push_file"
  owner (str, optional): Repository owner
  repo (str, optional): Repository name
  path (str, required): File path in the repository
  content (str, required): File content to write
  title (str, optional): Commit message (default: "Update {path}")
  branch (str, optional): Branch name (default: "main")
```

**Example**: `github(action="push_file", path="README.md", content="# My Project\nHello!", title="Update README")`

#### `list_actions` — List recent workflow runs

```
Parameters:
  action (str, required): "list_actions"
  owner (str, optional): Repository owner
  repo (str, optional): Repository name
```

**Example**: `github(action="list_actions")`

#### `trigger_action` — Trigger a workflow dispatch

```
Parameters:
  action (str, required): "trigger_action"
  owner (str, optional): Repository owner
  repo (str, optional): Repository name
  path (str, required): Workflow file name (e.g. "ci.yml")
  branch (str, optional): Branch to run against (default: "main")
```

**Example**: `github(action="trigger_action", path="deploy.yml", branch="release")`

## Managing Tools

```bash
# List all tools with tier and status
agent tools list

# Enable/disable a tool
agent tools enable shell_exec
agent tools disable browser_navigate
```

Via API:

```
GET  /api/v1/tools
PUT  /api/v1/tools/{name}/toggle  {"enabled": true}
```

## Custom Tools via Skills

Skills can register additional tools. See [Skills](skills.md) for how to create tools via the plugin system.
