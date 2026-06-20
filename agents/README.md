# Agent Workflow

## Sessions

| Session | Where | Role |
|---------|-------|------|
| Reviewer + Filter | repo root terminal | Receives natural language requests, translates to precise implementation, executes changes directly |
| Debugger | `agents/debugger/` terminal | Diagnoses pipeline errors from terminal output |

## Normal flow

    You (natural language) → Reviewer terminal
    ↓
    UNDERSTOOD: file / change / cache impact / scope
    ↓
    [executes edit directly]
    ↓
    Done. What changed and what to do next.

You see the translation step before anything is touched.
Stop the reviewer if it misread your intent — then redirect.

## Error flow

    Pipeline crashes or wrong output
    ↓
    Paste full terminal output → Debugger terminal
    ↓
    ROOT CAUSE + fix instruction
    ↓
    Bring fix back to Reviewer terminal → executed

## Opening terminals (VS Code)

Debugger opens automatically on folder open.
If it doesn't: Cmd+Shift+P → Tasks: Manage Automatic Tasks → Allow Automatic Tasks

Reviewer = the repo-root Claude session (open manually).
