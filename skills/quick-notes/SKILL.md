---
name: quick-notes
description: Save, search, and list quick markdown notes.
version: "0.1.0"
author: Agent Team
permissions:
  - safe
  - moderate
dependencies: []
triggers:
  - note
  - notes
  - remember
---

# Quick Notes

Save and retrieve quick markdown notes. Notes are stored as individual markdown files in `data/notes/`.

## Tools

- **save** — Save a new note with a title and content
- **search** — Search notes by keyword
- **list** — List all saved notes

## Usage

```
Save a note titled "Meeting" with content "Discussed project timeline"
Search my notes for "meeting"
List all my notes
```
