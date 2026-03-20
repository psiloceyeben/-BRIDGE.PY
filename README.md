```
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ████████╗██╗  ██╗███████╗                                  ║
    ║      ██╔══╝██║  ██║██╔════╝                                  ║
    ║      ██║   ███████║█████╗                                    ║
    ║      ██║   ██╔══██║██╔══╝                                    ║
    ║      ██║   ██║  ██║███████╗                                  ║
    ║      ╚═╝   ╚═╝  ╚═╝╚══════╝                                 ║
    ║   ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗               ║
    ║   ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝              ║
    ║   ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗                ║
    ║   ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝                ║
    ║   ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗              ║
    ║   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝             ║
    ║                                                              ║
    ║   cognitive architecture for artificial intelligence         ║
    ║   identity · routing · memory · habits · metabolism          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
```

# The Bridge

**One file. Any model. A complete cognitive body for AI.**

The industry spent $200B+ scaling models but never standardized what wraps around them. Every developer building on AI reinvents the same scaffolding: session management, memory, routing, state, identity, tools. The Bridge is a single Python file that solves this permanently.

Not a framework. A standard. The minimum viable body for artificial intelligence.

```
python bridge.py
```

That is the install. It auto-installs dependencies, walks you through vessel creation, and starts a fully cognitive AI system in under a minute.

---

## What It Is

A 2000-line Python file that gives any LLM a complete cognitive architecture:

```
bridge.py              <- the one file
vessel/
  VESSEL.md            <- identity (who the AI is)
  STATE.md             <- current state (heartbeat appends)
  HECATE.md            <- routing rules (optional, has defaults)
  habits.json          <- procedural memory (learns from use)
  hrr_habits.json      <- holographic memory vectors
  tree/                <- sephiroth node descriptions
    KETER.md ... MALKUTH.md
  vault/               <- declarative memory (Obsidian-compatible)
    INDEX.md
    sessions/
    knowledge/
    ideas/
    entities/
static/                <- web files (optional)
tools/                 <- drop-in plugin tools
.env                   <- API key + config
```

## The 12 Sections

| # | Section | What It Does |
|---|---------|-------------|
| 1 | **Imports & Config** | All paths relative. Env vars for everything. Zero hardcoded paths. |
| 2 | **Multi-LLM Provider** | Anthropic, OpenAI, Ollama, OpenRouter. Same interface. |
| 3 | **HRR Engine** | Holographic Reduced Representations. Sub-millisecond recall. No API cost. |
| 4 | **Vessel & Vault Init** | Creates identity + Obsidian-compatible knowledge graph on first run. |
| 5 | **Tree of Life / Hecate** | Every request routes through sephiroth. Not optional. |
| 6 | **Yesod Habits** | Procedural memory. Learns routes. Reinforcement. Forks on failure. |
| 7 | **Tools + Plugins** | 9 core tools + auto-discovery from tools/*.py. |
| 8 | **Operator Loop** | Agentic tool loop. Safe/dangerous split. Orphan repair. |
| 9 | **HTTP Endpoints** | /chat, /ask, /setup, /health, vault API. |
| 10 | **Auto-Init** | TTY wizard or browser setup. First run creates everything. |
| 11 | **Substrate** | Continuous memory transformation. Zero API cost. The unconscious. |
| 12 | **Heartbeat + Main** | Periodic pulse. Background metabolism. Uvicorn startup. |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/user/the-bridge.git
cd the-bridge

# 2. Set your API key
cp .env.example .env
# Edit .env with your key

# 3. Run
python bridge.py
```

First run in terminal:

```
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   THE BRIDGE — first run detected                        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝

  ◆ What should this vessel be called?
  > Atlas

  ◆ What is it for — and who is it for?
  > Infrastructure monitoring for our engineering team

  ◆ What voice or tone?
  > Direct, technical, no fluff

  ✓ Vessel "Atlas" created.
  Tree of Life nodes generated.
  Obsidian vault initialized.
  Starting server on port 8000...
```

You now have a cognitive AI with identity, routing, memory, habits, and metabolism.

---

## The Architecture

### Tree of Life (Not Optional)

Every request routes through the sephiroth — 10 nodes from KETER (intention) to MALKUTH (manifestation). This is not decoration. It forces structured transformation:

```
KETER ───── What does this request truly want?
   │
CHOKMAH ─── Raw creative approach
   │
BINAH ───── Structure and constraints
   │
 ... ────── Through specific paths with named qualities
   │
MALKUTH ─── Final output enters the world
```

HECATE classifies each request using the fast/cheap model and returns a route. Proven habits skip classification entirely — the system develops procedural memory through use.

### Holographic Memory (HRR)

Complex-valued vectors using circular convolution. Inspired by [NeoVertex1/nuggets](https://github.com/NeoVertex1/nuggets).

```python
hrr.bind("project_atlas", "monitoring infrastructure for 50 servers")
hrr.recall("atlas")  # retrieves the binding with similarity score
hrr.novelty("we discussed atlas monitoring")  # 0.15 (seen before)
hrr.novelty("new database migration plan")    # 0.92 (novel)
```

- **Sub-millisecond** recall from fixed-size vectors
- **Zero API cost** — pure numpy
- **Novelty detection** — filters vault commits, only saves what is new
- **Habit boosting** — HRR similarity improves habit matching

### Yesod Habits (Procedural Memory)

The system learns from every interaction:

```
Request: "deploy the staging server"
  -> Extract signature: [deploy, server, setup]
  -> Check habits: no match
  -> Route through Hecate: KETER -> GEVURAH -> MALKUTH
  -> Execute successfully
  -> Record success: confidence 0.33

Same request again:
  -> Habit match: confidence 0.67 (learning)
  -> Still routes through Hecate for verification

Third time:
  -> Habit proven: confidence 0.85
  -> Skips Hecate entirely — cached route
  -> Sub-millisecond routing
```

Habits fork on failure, get suspended at low confidence, and transfer between instances (copy the JSON).

### Substrate / Metabolism

**This is the part nobody else is building.**

A background loop runs every 5 minutes performing pure numpy transformations on the memory vectors:

- **Hebbian decay** — unused associations fade (0.98x), frequently recalled ones strengthen (1.02x)
- **Resonance amplification** — facts recalled 3+ times get boosted
- **Spectral consolidation** — FFT denoise, suppress weak frequency bins

Zero API cost. The vessel unconscious. Memory transforms between conversations the same way a human mind consolidates during rest.

### Plugin Tools

Drop a Python file in `tools/`:

```python
# tools/check_server.py

TOOL_DEFINITION = {
    "name": "check_server",
    "description": "Ping a server and return status.",
    "input_schema": {
        "type": "object",
        "properties": {"host": {"type": "string"}},
        "required": ["host"],
    },
}

SAFE = True  # auto-execute, no confirmation needed

def execute(inp: dict) -> str:
    import subprocess
    result = subprocess.run(["ping", "-c", "1", inp["host"]], capture_output=True, text=True)
    return "UP" if result.returncode == 0 else "DOWN"
```

Restart bridge.py. The tool appears in `/chat` automatically.

---

## Providers

```bash
# Anthropic (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
BRIDGE_MODEL=gpt-4.1
BRIDGE_MODEL_FAST=gpt-4.1-mini

# Ollama (local, free)
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
BRIDGE_MODEL=qwen2.5-coder:7b
BRIDGE_MODEL_FAST=qwen2.5:1.5b

# OpenRouter (any model)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
BRIDGE_MODEL=anthropic/claude-sonnet-4-6
BRIDGE_MODEL_FAST=anthropic/claude-haiku-4-5-20251001
```

Same bridge.py. Same architecture. Different brain.

---

## API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat` | POST | Token | Operator terminal — full tools, agentic loop |
| `/chat/confirm` | POST | Token | Approve/reject dangerous tool actions |
| `/chat/clear` | POST | Token | Reset session |
| `/ask` | POST | None | Visitor chat — no tools, 5 msg limit |
| `/setup` | GET | None | Browser wizard for vessel creation |
| `/setup` | POST | None | Process wizard form |
| `/health` | GET | None | Vessel status + HRR stats |
| `/vault/commit` | POST | None | Auto-commit conversation to vault via HRR |
| `/api/vault/list` | GET | Token | List vault notes |
| `/api/vault/read` | GET | Token | Read a vault note |
| `/api/vault/write` | POST | Token | Write a vault note |

---

## The Obsidian Vault

Every vessel maintains an Obsidian-compatible knowledge graph:

```
vault/
  INDEX.md                    <- central hub
  sessions/
    ATLAS_2025-03-19_1430.md  <- auto-committed conversation summaries
  knowledge/
    substrate_2025-03-19.md   <- metabolism transformation logs
    deployment_patterns.md    <- accumulated knowledge
  ideas/
    migration_plan.md         <- emerging plans
  entities/
    engineering_team.md       <- people and orgs
```

Notes use `[[wikilinks]]` to form a knowledge graph. Open the vault in Obsidian to see your AI mind as a visual network.

Conversations are committed to the vault when the HRR engine determines they contain novel information (novelty > 0.4). Redundant conversations are skipped. The system literally learns what is worth remembering.

---

## What This Replaces

| Before The Bridge | After |
|------------------|-------|
| Custom session management per project | `_chat_sessions` + `CONTEXT.md` |
| External vector database (Pinecone, etc.) | HRR engine — inline, numpy, 1024-dim |
| No memory between conversations | Obsidian vault + HRR novelty filter |
| No learning from past interactions | Yesod habits with reinforcement |
| Dead between requests | Substrate metabolism — continuous transformation |
| Model-locked architecture | 4 providers, same interface |
| Framework with 50 dependencies | One file. `python bridge.py` |

---

## Philosophy

Intelligence requires a body. Not a bigger brain — a body that has identity, that routes through structured transformation, that remembers, that develops habits through reinforcement, that metabolizes experience between conversations, that maintains a persistent knowledge graph.

The Bridge is that body. One file. Any model. Drop it in a folder and run it.

The AI industry will eventually standardize on something like this. This is the proposal.

---

## Credits

- **HRR Engine** inspired by [NeoVertex1/nuggets](https://github.com/NeoVertex1/nuggets)
- **Tree of Life routing** based on Kabbalistic sephiroth as a transformation framework
- Built with [Claude](https://claude.ai) and [The Grand Internet Hotel](https://thegrandinternethotel.com)

## License

MIT
