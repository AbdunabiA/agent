# Telegram Setup

Agent includes a Telegram bot powered by aiogram 3.x.

## Setup

### 1. Create a Bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the bot token

### 2. Get Your User ID

Message [@userinfobot](https://t.me/userinfobot) to get your Telegram user ID (a number like `123456789`).

### 3. Configure

Add to `.env`:

```
TELEGRAM_BOT_TOKEN=your-bot-token-here
```

Add to `agent.yaml`:

```yaml
channels:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    allowed_users: [123456789]  # Your Telegram user ID
```

**Security**: If `allowed_users` is empty, the bot accepts messages from everyone. Always set this in production.

### 4. Run

```bash
agent start
```

## Bot Commands

These commands are registered in the Telegram menu button:

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | Show available commands |
| `/new` | Start a new conversation |
| `/status` | Agent status |
| `/tools` | List available tools |
| `/session` | Current session info |
| `/soul` | View/edit agent personality |
| `/backend` | View/switch LLM backend |
| `/workdir` | View/change working directory |
| `/remind <delay> <text>` | Set a reminder (e.g. `/remind 30m Check the build`) |
| `/reminders` | List pending reminders |
| `/pause` | Pause message processing |
| `/resume` | Resume message processing |
| `/mute` | Disable heartbeat |
| `/unmute` | Enable heartbeat |

## Features

### Text Messages

Send any text message and the agent responds using the configured LLM.

### Voice Messages

Send a voice message and the agent processes it:
- **LLM native mode** (default): Audio is sent directly to the LLM (Claude/GPT-4o native audio support)
- **STT mode** (Whisper/Deepgram): Audio is transcribed first, then the transcription is processed as text

If voice reply is enabled, the agent responds with a synthesized voice message.

### Photos

Send a photo with an optional caption. The agent analyzes the image using multimodal LLM capabilities.

### Documents

Send a document (PDF, code files, etc.) and the agent reads and processes the content.

### Receiving Files, Images, and Videos

The agent can send files back to you! Just ask it to send a file and it will use the `send_file` tool:

- **Images** (`.jpg`, `.png`, `.gif`, `.webp`, `.bmp`) are sent as inline photos with preview
- **Videos** (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) are sent as inline videos with player
- **Other files** (PDFs, code, archives, etc.) are sent as document attachments

Examples:
- "Send me the config file at ~/project/config.yaml"
- "Share the screenshot you just took"
- "Send the video recording from /tmp/demo.mp4"

File size limit: 50 MB (Telegram bot API limit).

### Tool Approval

When the agent wants to use a moderate or dangerous tool, it sends an interactive approval message with Approve/Deny buttons. You can approve or deny each tool call directly in Telegram.

### Reminders

Set reminders directly in chat:
- Use the `/remind` command: `/remind 1h Check the deployment`
- Or just ask naturally: "Remind me in 30 minutes to review the PR"

The agent detects natural language reminder requests automatically.

## Architecture

```
Telegram User
     |
  aiogram 3.x
     |
  TelegramChannel
     |
  AgentLoop.process_message()
     |
  LLM + Tools + Memory
```

The Telegram channel:
- Maintains per-user sessions via SessionStore
- Respects the `allowed_users` allowlist
- Forwards events to the event bus
- Supports streaming responses (sent as message edits)
- Handles FILE_SEND events to deliver files back to users
