#!/usr/bin/env python3
"""
bridge.py — Universal cognitive architecture for AI.

A single file that gives any model a complete cognitive body:
identity, routing (Tree of Life), habits, memory (Obsidian vault + HRR),
metabolism (substrate), and tools.

Drop it in a folder, set an API key, run it.

Not a framework. A standard. The minimum viable body for artificial intelligence.
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 1: IMPORTS & CONFIG
# ╚════════════════════════════════════════════════════════════════════════════╝

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess as _subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

def _bootstrap_deps():
    """Install missing packages before importing them."""
    required = {"numpy": "numpy", "uvicorn": "uvicorn", "dotenv": "python-dotenv",
                "fastapi": "fastapi", "anthropic": "anthropic", "openai": "openai",
                "telegram": "python-telegram-bot"}
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        import subprocess as _sp
        print(f"\033[33m  Installing: {', '.join(missing)}...\033[0m")
        _sp.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
        print(f"\033[32m  ✓ Dependencies installed.\033[0m\n")

_bootstrap_deps()

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse

# ── paths: everything relative to where bridge.py lives ──
BASE_DIR    = Path(__file__).parent
ENV_FILE    = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

VESSEL_DIR   = Path(os.environ.get("VESSEL_DIR",  str(BASE_DIR / "vessel")))
STATIC_DIR   = Path(os.environ.get("STATIC_DIR",  str(BASE_DIR / "static")))
TOOLS_DIR    = Path(os.environ.get("TOOLS_DIR",   str(BASE_DIR / "tools")))
VAULT_DIR    = VESSEL_DIR / "vault"
SANDBOX_ROOT = BASE_DIR  # tools can only access files under here

# ── models ──
LLM_PROVIDER       = os.environ.get("LLM_PROVIDER", "anthropic").lower()
BRIDGE_MODEL       = os.environ.get("BRIDGE_MODEL",      "claude-sonnet-4-6")
BRIDGE_MODEL_FAST  = os.environ.get("BRIDGE_MODEL_FAST",  "claude-haiku-4-5-20251001")
BUILD_TOKEN        = os.environ.get("BUILD_TOKEN", "")
PORT               = int(os.environ.get("PORT", "8000"))

try:
    MAX_TOKENS = int(os.environ.get("BRIDGE_MAX_TOKENS", "16384"))
except (ValueError, TypeError):
    MAX_TOKENS = 16384

try:
    HEARTBEAT_INTERVAL = int(os.environ.get("BRIDGE_HEARTBEAT_MIN", "30")) * 60
except (ValueError, TypeError):
    HEARTBEAT_INTERVAL = 1800

try:
    SUBSTRATE_INTERVAL = int(os.environ.get("SUBSTRATE_INTERVAL", "150"))
except (ValueError, TypeError):
    SUBSTRATE_INTERVAL = 150

ALL_NODES = [
    "KETER", "CHOKMAH", "BINAH", "CHESED", "GEVURAH",
    "TIFERET", "NETZACH", "HOD", "YESOD", "MALKUTH",
]

DEFAULT_ROUTE = {
    "nodes": ["KETER", "TIFERET", "MALKUTH"],
    "transitions": [
        {"from": "KETER",   "to": "TIFERET", "path": "GIMEL", "quality": "long intuitive crossing — what is hidden becomes central"},
        {"from": "TIFERET", "to": "MALKUTH", "path": "TAV",   "quality": "complete integration — all memory arrives whole in the world"},
    ],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRIDGE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge")

app = FastAPI()


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 2: MULTI-LLM PROVIDER (Anthropic, OpenAI, Ollama, OpenRouter)
# ╚════════════════════════════════════════════════════════════════════════════╝

OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

if LLM_PROVIDER == "anthropic":
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=1)

elif LLM_PROVIDER in ("openai", "ollama", "openrouter"):
    from openai import OpenAI as _OpenAI

    if LLM_PROVIDER == "ollama":
        _oai_client = _OpenAI(base_url=OLLAMA_HOST + "/v1", api_key="ollama")
    elif LLM_PROVIDER == "openrouter":
        _oai_client = _OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY or os.environ.get("OPENROUTER_API_KEY", ""),
        )
    else:
        _oai_client = _OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    _MODEL_MAP = {
        "openai": {
            "main":  os.environ.get("BRIDGE_MODEL",      "gpt-4.1"),
            "fast":  os.environ.get("BRIDGE_MODEL_FAST",  "gpt-4.1-mini"),
        },
        "ollama": {
            "main":  os.environ.get("BRIDGE_MODEL",      "qwen2.5-coder:7b"),
            "fast":  os.environ.get("BRIDGE_MODEL_FAST",  "qwen2.5:1.5b"),
        },
        "openrouter": {
            "main":  os.environ.get("BRIDGE_MODEL",      "anthropic/claude-sonnet-4-6"),
            "fast":  os.environ.get("BRIDGE_MODEL_FAST",  "anthropic/claude-haiku-4-5-20251001"),
        },
    }

    def _resolve_model(anthropic_model: str) -> str:
        m = anthropic_model.lower()
        if "haiku" in m or "fast" in m or "classify" in m:
            return _MODEL_MAP[LLM_PROVIDER]["fast"]
        return _MODEL_MAP[LLM_PROVIDER]["main"]

    def _convert_tools_to_openai(tools: list) -> list:
        if not tools:
            return []
        return [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        } for t in tools]

    def _convert_messages_to_openai(system: str, messages: list) -> list:
        oai_msgs = []
        if system:
            oai_msgs.append({"role": "system", "content": system})
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                oai_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts, tool_calls, tool_results = [], [], []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            text_parts.append(block.get("text", ""))
                        elif btype == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            })
                        elif btype == "tool_result":
                            rc = block.get("content", "")
                            if isinstance(rc, list):
                                rc = " ".join(b.get("text", "") for b in rc if isinstance(b, dict)) or str(rc)
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(rc),
                            })
                    elif hasattr(block, "type"):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {"name": block.name, "arguments": json.dumps(block.input)},
                            })
                if role == "user" and tool_results:
                    if text_parts:
                        oai_msgs.append({"role": "user", "content": "\n".join(text_parts)})
                    oai_msgs.extend(tool_results)
                elif role == "assistant" and tool_calls:
                    msg = {"role": "assistant", "tool_calls": tool_calls}
                    if text_parts:
                        msg["content"] = "\n".join(text_parts)
                    oai_msgs.append(msg)
                else:
                    oai_msgs.append({"role": role, "content": "\n".join(text_parts) or ""})
        return oai_msgs

    class _AnthropicShim:
        """Wraps OpenAI-compatible clients to return Anthropic-format responses."""

        class messages:
            @staticmethod
            def create(*, model="", max_tokens=4096, system="", messages=None,
                       tools=None, timeout=120, **kwargs):
                actual_model = _resolve_model(model)
                oai_msgs = _convert_messages_to_openai(system, messages or [])
                oai_tools = _convert_tools_to_openai(tools) if tools else None

                call_kwargs = {
                    "model": actual_model,
                    "max_tokens": max_tokens,
                    "messages": oai_msgs,
                    "timeout": timeout,
                }
                if oai_tools:
                    call_kwargs["tools"] = oai_tools

                resp = _oai_client.chat.completions.create(**call_kwargs)
                choice = resp.choices[0]

                content_blocks = []
                if choice.message.content:
                    content_blocks.append(type("TextBlock", (), {
                        "type": "text", "text": choice.message.content,
                    })())
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        try:
                            inp = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            inp = {}
                        content_blocks.append(type("ToolUseBlock", (), {
                            "type": "tool_use", "id": tc.id,
                            "name": tc.function.name, "input": inp,
                        })())

                stop = "end_turn"
                if choice.finish_reason == "tool_calls":
                    stop = "tool_use"
                elif choice.finish_reason == "length":
                    stop = "max_tokens"

                return type("Response", (), {
                    "content": content_blocks,
                    "stop_reason": stop,
                    "model": actual_model,
                    "usage": type("Usage", (), {
                        "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                        "output_tokens": getattr(resp.usage, "completion_tokens", 0),
                    })(),
                })()

    client = _AnthropicShim()

else:
    # Fallback to Anthropic
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=1)
    LLM_PROVIDER = "anthropic"


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 3: HRR ENGINE (Holographic Reduced Representations)
# ╚════════════════════════════════════════════════════════════════════════════╝

HRR_DIM = 1024


def _seed_vector(label: str, dim: int = HRR_DIM) -> np.ndarray:
    """Deterministic unit-length complex vector from a string seed."""
    h = int(hashlib.sha256(label.encode()).hexdigest(), 16)
    rng = np.random.RandomState(h % (2**31))
    phases = rng.uniform(0, 2 * np.pi, dim)
    return np.exp(1j * phases) / np.sqrt(dim)


def _circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution — the binding operation."""
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))


def _circular_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular correlation — the unbinding (recall) operation."""
    return np.fft.ifft(np.conj(np.fft.fft(a)) * np.fft.fft(b))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two complex vectors."""
    return float(np.abs(np.dot(a.conj(), b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "like",
    "through", "after", "over", "between", "out", "up", "i", "you", "we",
    "they", "he", "she", "it", "me", "my", "your", "our", "this", "that",
    "these", "those", "what", "which", "who", "how", "when", "where", "why",
    "not", "no", "yes", "just", "also", "very", "so", "and", "but", "or",
    "if", "then", "than", "too", "its", "im", "dont", "thats", "ive",
    "lets", "want", "need", "think",
})


class HolographicMemory:
    """Holographic Reduced Representation memory — one per vessel."""

    def __init__(self, path: str = None, dim: int = HRR_DIM):
        self.dim = dim
        self.path = Path(path) if path else None
        self.memory = np.zeros(dim, dtype=complex)
        self.index: list[dict] = []
        self.recall_counts: dict[str, int] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self):
        try:
            data = json.loads(self.path.read_text())
            self.index = data.get("index", [])
            self.recall_counts = data.get("recall_counts", {})
            self.memory = np.zeros(self.dim, dtype=complex)
            for entry in self.index:
                kv = _seed_vector(entry["key"], self.dim)
                vv = _seed_vector(entry["value"], self.dim)
                self.memory += _circular_conv(kv, vv)
            log.info(f"HRR loaded {len(self.index)} facts from {self.path}")
        except Exception as e:
            log.warning(f"HRR load error: {e}")
            self.memory = np.zeros(self.dim, dtype=complex)
            self.index = []

    def _save(self):
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1, "dim": self.dim,
                "index": self.index, "recall_counts": self.recall_counts,
            }
            self.path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            log.warning(f"HRR save error: {e}")

    def bind(self, key: str, value: str, metadata: dict = None):
        """Encode a key-value fact into the holographic memory."""
        kv = _seed_vector(key, self.dim)
        vv = _seed_vector(value, self.dim)
        self.memory += _circular_conv(kv, vv)
        entry = {"key": key, "value": value[:500]}
        if metadata:
            entry["meta"] = metadata
        self.index.append(entry)
        self._save()

    def recall(self, key: str) -> list[tuple]:
        """Recall values associated with a key. Returns [(value, similarity, meta)]."""
        if np.linalg.norm(self.memory) < 1e-10:
            return []
        kv = _seed_vector(key, self.dim)
        retrieved = _circular_corr(kv, self.memory)
        results = []
        for entry in self.index:
            vv = _seed_vector(entry["value"], self.dim)
            sim = _cosine_sim(retrieved, vv)
            if sim > 0.05:
                results.append((entry["value"], sim, entry.get("meta", {})))
        self.recall_counts[key] = self.recall_counts.get(key, 0) + 1
        self._save()
        return sorted(results, key=lambda x: x[1], reverse=True)[:10]

    def forget(self, key: str, value: str):
        """Remove a specific binding from the superposition."""
        kv = _seed_vector(key, self.dim)
        vv = _seed_vector(value, self.dim)
        self.memory -= _circular_conv(kv, vv)
        self.index = [e for e in self.index if not (e["key"] == key and e["value"] == value)]
        self._save()

    def novelty(self, text: str) -> float:
        """How novel is this text? 0.0 = redundant, 1.0 = completely new."""
        if not self.index:
            return 1.0
        words = set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()) - _STOP_WORDS
        if not words:
            return 0.0
        max_overlap = 0.0
        for entry in self.index:
            fact_words = set(re.sub(r"[^a-z0-9 ]", " ",
                (entry.get("key", "") + " " + entry.get("value", "")).lower()
            ).split()) - _STOP_WORDS
            if not fact_words:
                continue
            jaccard = len(words & fact_words) / len(words | fact_words)
            if jaccard > max_overlap:
                max_overlap = jaccard
        query_vec = np.zeros(self.dim, dtype=complex)
        for w in list(words)[:20]:
            query_vec += _seed_vector(w, self.dim)
        hrr_sim = 0.0
        if np.linalg.norm(query_vec) > 1e-10:
            hrr_sim = _cosine_sim(query_vec / np.linalg.norm(query_vec), self.memory)
        combined = max_overlap * 0.7 + float(hrr_sim) * 0.3
        return float(max(0.0, min(1.0, 1.0 - combined)))

    def get_hot_facts(self, threshold: int = 3) -> list[dict]:
        """Facts recalled more than threshold times."""
        return [{"key": k, "recall_count": c} for k, c in self.recall_counts.items() if c >= threshold]

    def stats(self) -> dict:
        return {
            "total_facts": len(self.index),
            "vector_norm": float(np.linalg.norm(self.memory)),
            "hot_facts": len(self.get_hot_facts()),
            "unique_keys": len(set(e["key"] for e in self.index)),
        }


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 4: VESSEL & VAULT INIT
# ╚════════════════════════════════════════════════════════════════════════════╝

def _init_dirs():
    """Create vessel/, vault/, static/, tools/ on first run."""
    VESSEL_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    (VESSEL_DIR / "tree").mkdir(exist_ok=True)


def _init_vault():
    """Create vault directory structure and INDEX.md."""
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("sessions", "knowledge", "ideas", "entities"):
        (VAULT_DIR / sub).mkdir(exist_ok=True)

    index = VAULT_DIR / "INDEX.md"
    if not index.exists():
        vessel_name = "VESSEL"
        vmd = VESSEL_DIR / "VESSEL.md"
        if vmd.exists():
            for line in vmd.read_text(errors="replace").split("\n"):
                if line.startswith("# "):
                    vessel_name = line[2:].strip()
                    break
        index.write_text(
            f"# {vessel_name} — Knowledge Vault\n\n"
            "This is the central knowledge graph for this vessel.\n\n"
            "## Structure\n"
            "- [[sessions/]] — Conversation summaries and decisions\n"
            "- [[knowledge/]] — Accumulated knowledge and references\n"
            "- [[ideas/]] — Ideas, plans, and future directions\n"
            "- [[entities/]] — People, orgs, and external references\n\n"
            "## Recent\n"
            "*(notes will appear here as the vault grows)*\n"
        )
        log.info("VAULT initialized at " + str(VAULT_DIR))


def read(path: Path) -> str:
    """Read a file, return empty string if missing."""
    return path.read_text(errors="replace").strip() if path.exists() else ""


def load_vessel() -> dict:
    """Load all vessel context files."""
    return {
        "vessel":  read(VESSEL_DIR / "VESSEL.md"),
        "state":   read(VESSEL_DIR / "STATE.md"),
        "hecate":  read(VESSEL_DIR / "HECATE.md"),
        "malkuth": read(VESSEL_DIR / "tree" / "MALKUTH.md"),
    }


def load_node(name: str) -> str:
    """Load a single Tree of Life node description."""
    return read(VESSEL_DIR / "tree" / f"{name.upper()}.md")


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 5: TREE OF LIFE / HECATE (path-aware classifier)
# ╚════════════════════════════════════════════════════════════════════════════╝

def build_tree_context(route: dict) -> str:
    """Assemble node descriptions + path qualities for a given route."""
    nodes = route["nodes"]
    transitions = {
        (t["from"], t["to"]): (t["path"], t["quality"])
        for t in route.get("transitions", [])
    }
    sections = []
    for i, node in enumerate(nodes):
        node_text = load_node(node)
        if i > 0:
            prev = nodes[i - 1]
            if (prev, node) in transitions:
                path_name, quality = transitions[(prev, node)]
                sections.append(
                    f"── PATH {path_name} ({prev} → {node}) ──\n"
                    f"Transformation as you cross: {quality}\n"
                )
        if node_text and node != "MALKUTH":
            sections.append(f"## {node}\n{node_text}")
    return "\n\n".join(sections)


def _repair_json(raw: str) -> str:
    """Clean up LLM JSON: strip markdown fences, trailing commas, comments."""
    raw = re.sub(r'^```json\s*', '', raw.strip())
    raw = re.sub(r'```\s*$', '', raw.strip())
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return "\n".join(ln for ln in raw.splitlines() if not ln.strip().startswith('//')).strip()


def _get_text(resp) -> str:
    """Safely extract text from an API response."""
    if resp and hasattr(resp, "content") and resp.content:
        for block in resp.content:
            if hasattr(block, "text") and block.text:
                return block.text.strip()
    return ""


def hecate(ctx: dict, request_text: str) -> dict:
    """
    HECATE reads the request and returns a route through the sephiroth tree.
    Checks habits first — proven habits skip classification entirely.
    Uses the fast model. Falls back to DEFAULT_ROUTE on failure.
    """
    habits = load_habits()

    # Check blacklist
    blacklisted = check_blacklist(request_text, habits)
    if blacklisted:
        log.info("HECATE: avoiding blacklisted route: %s", blacklisted.get("failure_mode", "unknown"))

    # Proven habit — skip classification
    habit_key, habit = match_habit(request_text, habits)
    if habit and habit.get("status") == "proven":
        log.info("HECATE: proven habit %s (conf=%.2f)", habit_key, habit.get("confidence", 0))
        return {"nodes": habit["path"], "transitions": [], "_habit": True, "_habit_key": habit_key}

    # No HECATE.md = no routing rules
    if not ctx.get("hecate"):
        return DEFAULT_ROUTE

    system = f"""{ctx['hecate']}

You are HECATE. Read the request, apply the routing rules, resolve the path
for each consecutive node pair using the PATH LOOKUP TABLE, and return the
route as valid JSON.

Respond with ONLY the JSON object. No markdown. No explanation."""

    prompt = f"VESSEL:\n{ctx['vessel'][:500]}\n\nREQUEST: {request_text}\n\nReturn the route JSON."

    try:
        resp = client.messages.create(
            model=BRIDGE_MODEL_FAST,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            timeout=180,
        )
        raw = resp.content[0].text.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON in: {raw!r}")
        route = json.loads(_repair_json(match.group()))
        nodes = [n for n in route.get("nodes", []) if n in ALL_NODES]
        if not nodes:
            raise ValueError("empty node list")
        if nodes[-1] != "MALKUTH":
            nodes = [n for n in nodes if n != "MALKUTH"] + ["MALKUTH"]
        transitions = route.get("transitions", [])
        log.info("HECATE route: " + " → ".join(f"{t['from']}[{t['path']}]→{t['to']}" for t in transitions))
        return {"nodes": nodes, "transitions": transitions}
    except Exception as e:
        log.warning(f"HECATE fallback ({e})")
        return DEFAULT_ROUTE


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 6: YESOD HABITS (procedural memory)
# ╚════════════════════════════════════════════════════════════════════════════╝

HABITS_FILE = VESSEL_DIR / "habits.json"
_hrr_habits = HolographicMemory(path=str(VESSEL_DIR / "hrr_habits.json"))

TASK_VOCAB = {
    "build", "create", "add", "delete", "remove", "edit", "update", "change",
    "fix", "debug", "deploy", "publish", "setup", "configure", "install",
    "write", "read", "search", "find", "list", "help", "explain", "analyze",
    "connect", "send", "receive", "schedule", "monitor", "alert", "notify",
    "page", "site", "file", "data", "api", "server", "database", "service",
    "user", "account", "auth", "login", "register", "profile", "settings",
    "style", "theme", "layout", "design", "image", "upload", "download",
    "test", "check", "verify", "validate", "review", "approve", "reject",
    "start", "stop", "restart", "status", "log", "report", "summary",
}

CONFIDENCE_THRESHOLD = 0.6


def load_habits() -> dict:
    if HABITS_FILE.exists():
        try:
            return json.loads(HABITS_FILE.read_text())
        except Exception:
            pass
    return {"version": 1, "routes": {}, "blacklist": {}}


def save_habits(habits: dict):
    try:
        HABITS_FILE.write_text(json.dumps(habits, indent=2))
    except Exception as e:
        log.warning(f"Could not save habits: {e}")


def _extract_signature(text: str) -> list:
    words = set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())
    return sorted(words & TASK_VOCAB)


def _make_task_key(text: str) -> str:
    sig = _extract_signature(text)
    return "_".join(sig[:4]) if sig else "unknown"


def _calc_confidence(habit: dict) -> float:
    """Weighted moving average: recent results count 2x."""
    recent = habit.get("recent", [])[-10:]
    total_s = habit.get("successes", 0)
    old_s = max(total_s - sum(recent), 0)
    old_f = max(habit.get("failures", 0) - recent.count(0), 0)
    old_total = old_s + old_f
    recent_s = sum(recent)
    recent_total = len(recent)
    if recent_total + old_total == 0:
        return 0.0
    return min(1.0, (recent_s * 2 + old_s) / (recent_total * 2 + max(old_total, 1)))


def _check_conditions(conditions: dict, context: dict) -> bool:
    for key, expr in conditions.items():
        val = context.get(key)
        if val is None:
            continue
        try:
            if isinstance(expr, str) and expr.startswith("<") and not (int(val) < int(expr[1:])):
                return False
            if isinstance(expr, str) and expr.startswith(">=") and not (int(val) >= int(expr[2:])):
                return False
        except (ValueError, TypeError):
            continue
    return True


def match_habit(request_text: str, habits: dict, context: dict = None) -> tuple:
    """Find best matching habit. Returns (key, habit) or (None, None)."""
    words = set(re.sub(r"[^a-z0-9 ]", " ", request_text.lower()).split())
    best_key, best_habit, best_score = None, None, 0.0

    try:
        n = _hrr_habits.novelty(request_text)
        if n < 0.3:
            log.info(f"HABIT HRR: strong match (novelty={n:.2f})")
    except Exception:
        pass

    for key, habit in habits.get("routes", {}).items():
        if habit.get("status") not in ("proven", "learning"):
            continue
        if habit.get("confidence", 0) < CONFIDENCE_THRESHOLD:
            continue
        sig_words = set(habit.get("signature", []))
        overlap = len(words & sig_words)
        if overlap < 2:
            continue
        score = overlap * habit.get("confidence", 0.5)
        try:
            ht = " ".join(habit.get("signature", [])) + " " + " ".join(habit.get("path", []))
            score += (1.0 - _hrr_habits.novelty(request_text + " " + ht)) * 0.3
        except Exception:
            pass
        if habit.get("conditions") and context and not _check_conditions(habit["conditions"], context):
            continue
        if habit.get("parent"):
            score *= 1.2
        if score > best_score:
            best_score, best_key, best_habit = score, key, habit

    return best_key, best_habit


def check_blacklist(request_text: str, habits: dict) -> dict | None:
    words = set(re.sub(r"[^a-z0-9 ]", " ", request_text.lower()).split())
    for key, entry in habits.get("blacklist", {}).items():
        if len(words & set(entry.get("signature", []))) >= 2:
            return entry
    return None


def record_success(habits: dict, task_key: str, signature: list, path: list):
    if task_key not in habits["routes"]:
        habits["routes"][task_key] = {
            "signature": signature, "path": path, "confidence": 0.0,
            "successes": 0, "failures": 0, "recent": [],
            "status": "learning", "conditions": {}, "forks": [],
        }
    h = habits["routes"][task_key]
    h["successes"] = h.get("successes", 0) + 1
    h["recent"] = (h.get("recent", []) + [1])[-10:]
    h["last_used"] = datetime.now(timezone.utc).isoformat()
    h["confidence"] = _calc_confidence(h)
    if h.get("path") != path:
        h["path"] = path
    if h["successes"] >= 3 and h["confidence"] >= 0.7:
        h["status"] = "proven"
    save_habits(habits)
    try:
        _hrr_habits.bind(f"success_{task_key}", " ".join(signature) + " " + " ".join(path),
                         metadata={"confidence": h["confidence"], "status": h["status"]})
    except Exception:
        pass


def record_failure(habits: dict, task_key: str, signature: list, path: list,
                   failure_reason: str, context: dict = None):
    if task_key in habits.get("routes", {}):
        h = habits["routes"][task_key]
        h["failures"] = h.get("failures", 0) + 1
        h["recent"] = (h.get("recent", []) + [0])[-10:]
        h["confidence"] = _calc_confidence(h)
        if h["confidence"] < 0.3:
            h["status"] = "suspended"
        elif h["confidence"] < 0.6 and h.get("status") == "proven":
            h["status"] = "learning"
        if context and h.get("status") != "suspended":
            fork_key = task_key + "_fork_" + str(len(h.get("forks", [])))
            h.setdefault("forks", []).append(fork_key)
            habits["routes"][fork_key] = {
                "signature": signature + _extract_signature(failure_reason),
                "path": path, "confidence": 0.0, "successes": 0, "failures": 0,
                "recent": [], "status": "learning", "conditions": context,
                "parent": task_key, "forks": [], "assessment": failure_reason,
            }
    else:
        habits.setdefault("blacklist", {})[task_key] = {
            "signature": signature, "failed_path": path,
            "failure_mode": failure_reason,
            "recorded": datetime.now(timezone.utc).isoformat(),
        }
    save_habits(habits)
    try:
        _hrr_habits.bind(f"failure_{task_key}",
                         " ".join(signature) + " FAIL " + failure_reason[:100],
                         metadata={"type": "failure"})
    except Exception:
        pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 7: TOOLS + PLUGIN LOADER
# ╚════════════════════════════════════════════════════════════════════════════╝

OPERATOR_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories at a path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Requires confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":        {"type": "string", "description": "File path"},
                "content":     {"type": "string", "description": "Full file content"},
                "description": {"type": "string", "description": "What this change does"},
            },
            "required": ["path", "content", "description"],
        },
    },
    {
        "name": "edit_file",
        "description": "Targeted search-and-replace edit. Requires confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":        {"type": "string", "description": "File path"},
                "old_text":    {"type": "string", "description": "Exact text to find (must be unique)"},
                "new_text":    {"type": "string", "description": "Replacement text"},
                "description": {"type": "string", "description": "What this change does"},
            },
            "required": ["path", "old_text", "new_text", "description"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command. Requires confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command":     {"type": "string", "description": "Shell command"},
                "description": {"type": "string", "description": "What this does"},
            },
            "required": ["command", "description"],
        },
    },
    {
        "name": "vault_write",
        "description": "Write a note to the Obsidian vault. Use [[wikilinks]].",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Note path (e.g. knowledge/topic.md)"},
                "content": {"type": "string", "description": "Markdown with [[wikilinks]]"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "vault_read",
        "description": "Read a vault note.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Note path"}},
            "required": ["path"],
        },
    },
    {
        "name": "vault_list",
        "description": "List vault notes.",
        "input_schema": {
            "type": "object",
            "properties": {"folder": {"type": "string", "description": "Subfolder (empty=all)"}},
        },
    },
    {
        "name": "vault_search",
        "description": "Search vault notes.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search text"}},
            "required": ["query"],
        },
    },
]

SAFE_TOOLS = {"read_file", "list_dir", "vault_read", "vault_list", "vault_search"}
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command", "vault_write"}

# ── plugin loader ──

_plugin_tools = []


def _load_plugins():
    """Scan tools/*.py for plugin tools. Each exports TOOL_DEFINITION, SAFE, execute(inp)."""
    global _plugin_tools
    # Remove previously loaded plugins from OPERATOR_TOOLS to avoid duplicates
    existing_plugin_names = {p["definition"]["name"] for p in _plugin_tools}
    for i in range(len(OPERATOR_TOOLS) - 1, -1, -1):
        if OPERATOR_TOOLS[i]["name"] in existing_plugin_names:
            OPERATOR_TOOLS.pop(i)
    _plugin_tools = []
    if not TOOLS_DIR.exists():
        return
    for py_file in sorted(TOOLS_DIR.glob("*.py")):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "TOOL_DEFINITION") and hasattr(mod, "execute"):
                defn = mod.TOOL_DEFINITION
                is_safe = getattr(mod, "SAFE", False)
                _plugin_tools.append({"definition": defn, "execute": mod.execute, "safe": is_safe})
                if defn["name"] not in {t["name"] for t in OPERATOR_TOOLS}:
                    OPERATOR_TOOLS.append(defn)
                if is_safe:
                    SAFE_TOOLS.add(defn["name"])
                else:
                    DANGEROUS_TOOLS.add(defn["name"])
                log.info(f"PLUGIN loaded: {defn['name']} ({'safe' if is_safe else 'dangerous'})")
        except Exception as e:
            log.warning(f"PLUGIN error {py_file.name}: {e}")


def _resolve_path(raw: str) -> Path:
    """Resolve a path, making relative paths relative to BASE_DIR."""
    p = Path(raw)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


def _exec_safe_tool(name: str, inp: dict) -> str:
    """Execute read-only tools immediately."""
    try:
        if name == "read_file":
            p = _resolve_path(inp["path"])
            if not (p == SANDBOX_ROOT or str(p).startswith(str(SANDBOX_ROOT) + os.sep)):
                return "Access denied: path outside sandbox"
            if not p.exists():
                return f"File not found: {inp['path']}"
            text = p.read_text(errors="replace")
            return text[:50000] + f"\n... (truncated — {len(text)} chars)" if len(text) > 50000 else text

        if name == "list_dir":
            p = _resolve_path(inp["path"])
            if not (p == SANDBOX_ROOT or str(p).startswith(str(SANDBOX_ROOT) + os.sep)):
                return "Access denied: path outside sandbox"
            if not p.exists():
                return f"Not found: {inp['path']}"
            rows = []
            for item in sorted(p.iterdir()):
                tag = "dir " if item.is_dir() else "file"
                size = f"  {item.stat().st_size}b" if item.is_file() else ""
                rows.append(f"{tag}  {item.name}{size}")
            return "\n".join(rows) or "(empty)"

        if name == "vault_read":
            p = (VAULT_DIR / inp["path"]).resolve()
            if not str(p).startswith(str(VAULT_DIR.resolve())):
                return "Access denied"
            if not p.exists():
                return f"Note not found: {inp['path']}"
            text = p.read_text(errors="replace")
            return text[:20000] + "\n... (truncated)" if len(text) > 20000 else text

        if name == "vault_list":
            folder = inp.get("folder", "").strip("/")
            vdir = VAULT_DIR / folder if folder else VAULT_DIR
            if not vdir.exists():
                return "(vault empty)"
            rows = []
            for md in sorted(vdir.rglob("*.md")):
                rel = str(md.relative_to(VAULT_DIR))
                first = md.read_text(errors="replace").split("\n")[0][:80] if md.exists() else ""
                rows.append(f"{rel}  |  {first}")
            return "\n".join(rows) or "(no notes)"

        if name == "vault_search":
            q = inp["query"].lower()
            results = []
            if not VAULT_DIR.exists():
                return "(vault empty)"
            for md in sorted(VAULT_DIR.rglob("*.md")):
                try:
                    lines = md.read_text(errors="replace").split("\n")
                    hits = [(i+1, l.strip()) for i, l in enumerate(lines) if q in l.lower()]
                    if hits:
                        rel = str(md.relative_to(VAULT_DIR))
                        for ln, text in hits[:3]:
                            results.append(f"{rel}:{ln}  {text[:100]}")
                except Exception:
                    pass
            return "\n".join(results[:30]) or "No matches"

        # Check plugins
        for plugin in _plugin_tools:
            if plugin["definition"]["name"] == name and plugin["safe"]:
                return str(plugin["execute"](inp))

    except Exception as e:
        return f"Error: {e}"
    return "Unknown tool"


def _exec_dangerous_tool(name: str, inp: dict) -> str:
    """Execute write/run tools after confirmation."""
    try:
        if name == "write_file":
            p = _resolve_path(inp["path"])
            if not (p == SANDBOX_ROOT or str(p).startswith(str(SANDBOX_ROOT) + os.sep)):
                return "Access denied"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inp["content"])
            return f"Written {len(inp['content'])} chars → {inp['path']}"

        if name == "edit_file":
            p = _resolve_path(inp["path"])
            if not (p == SANDBOX_ROOT or str(p).startswith(str(SANDBOX_ROOT) + os.sep)):
                return "Access denied"
            if not p.exists():
                return f"File not found: {inp['path']}"
            text = p.read_text(errors="replace")
            old, new = inp["old_text"], inp["new_text"]
            count = text.count(old)
            if count == 0:
                return f"old_text not found in {inp['path']}"
            if count > 1:
                return f"old_text found {count} times — must be unique"
            p.write_text(text.replace(old, new, 1))
            return f"Edited {inp['path']}: replaced {len(old)} → {len(new)} chars"

        if name == "run_command":
            result = _subprocess.run(
                inp["command"], shell=True, capture_output=True, text=True,
                timeout=180, cwd=str(BASE_DIR),
            )
            out = (result.stdout + result.stderr).strip()
            return (out[:3000] + "\n... (truncated)") if len(out) > 3000 else (out or "(no output)")

        if name == "vault_write":
            p = (VAULT_DIR / inp["path"]).resolve()
            if not str(p).startswith(str(VAULT_DIR.resolve())):
                return "Access denied"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inp["content"])
            # Bind to HRR so substrate can transform it
            try:
                hrr = HolographicMemory(path=str(VAULT_DIR / "hrr_memory.json"))
                hrr.bind(f"vault_{inp['path']}", inp['content'][:500],
                         metadata={"type": "vault_write", "path": inp["path"]})
                log.info(f"VAULT wrote + HRR bound {len(inp['content'])} chars → {inp['path']}")
            except Exception as e:
                log.warning(f"VAULT wrote but HRR bind failed: {e}")
                log.info(f"VAULT wrote {len(inp['content'])} chars → {inp['path']}")
            return f"Saved: {inp['path']} ({len(inp['content'])} chars)"

        for plugin in _plugin_tools:
            if plugin["definition"]["name"] == name and not plugin["safe"]:
                return str(plugin["execute"](inp))

    except _subprocess.TimeoutExpired:
        return "Command timed out (180s)"
    except Exception as e:
        return f"Error: {e}"
    return "Unknown tool"


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 8: OPERATOR LOOP + HISTORY
# ╚════════════════════════════════════════════════════════════════════════════╝

_chat_sessions: dict = {}
_chat_pending: dict = {}

CHAT_HISTORY_FILE = VESSEL_DIR / "chat_history.json"
CHAT_CONTEXT_FILE = VESSEL_DIR / "CONTEXT.md"
CHAT_HISTORY_MAX  = 100
CHAT_HISTORY_KEEP = 20


def _trim_history(history: list):
    """Trim large tool content to prevent context bloat."""
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        items = msg.get("content", [])
        if not isinstance(items, list):
            continue
        for block in items:
            if not isinstance(block, dict):
                continue
            if role == "assistant" and block.get("type") == "tool_use":
                inp = block.get("input", {})
                if block.get("name") == "write_file" and isinstance(inp.get("content"), str) and len(inp["content"]) > 300:
                    inp["content"] = f"(wrote {len(inp['content'])} chars to {inp.get('path', '?')})"
                if block.get("name") == "edit_file":
                    for k in ("old_text", "new_text"):
                        if isinstance(inp.get(k), str) and len(inp[k]) > 200:
                            inp[k] = inp[k][:100] + "...(trimmed)"
            if role == "user" and block.get("type") == "tool_result":
                if isinstance(block.get("content"), str) and len(block["content"]) > 500:
                    block["content"] = block["content"][:500] + " ... (trimmed)"


async def _summarize_and_compress(session_id: str, history: list, vessel_text: str) -> list:
    """Summarize older messages into CONTEXT.md, keep last CHAT_HISTORY_KEEP."""
    older = history[:-CHAT_HISTORY_KEEP]
    recent = history[-CHAT_HISTORY_KEEP:]

    lines = []
    for msg in older:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
        lines.append(f"{msg.get('role', '').upper()}: {content}")

    existing = CHAT_CONTEXT_FILE.read_text().strip() if CHAT_CONTEXT_FILE.exists() else ""
    prompt = (
        f"Summarize this conversation for vessel context.\nVessel: {vessel_text[:400]}\n\n"
        + (f"Existing context:\n{existing}\n\n" if existing else "")
        + "New conversation:\n" + "\n".join(lines)
        + "\n\nConcise running summary, 3-6 sentences."
    )

    try:
        resp = await asyncio.to_thread(lambda: client.messages.create(
            model=BRIDGE_MODEL, max_tokens=512,
            messages=[{"role": "user", "content": prompt}], timeout=60,
        ))
        CHAT_CONTEXT_FILE.write_text("# Operator Context\n\n" + resp.content[0].text.strip() + "\n")
        log.info(f"CHAT summarized {len(older)} msgs → CONTEXT.md")
    except Exception as e:
        log.warning(f"Summarization failed: {e}")

    return recent


def _load_chat_history(session_id: str) -> list:
    try:
        if CHAT_HISTORY_FILE.exists():
            return json.loads(CHAT_HISTORY_FILE.read_text()).get(session_id, [])
    except Exception:
        pass
    return []


def _save_chat_history(session_id: str, history: list):
    try:
        data = {}
        if CHAT_HISTORY_FILE.exists():
            try:
                data = json.loads(CHAT_HISTORY_FILE.read_text())
            except Exception:
                pass
        data[session_id] = history[-CHAT_HISTORY_MAX:]
        CHAT_HISTORY_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning(f"History save error: {e}")


def _serialize_block(b):
    """Serialize an API response content block to a dict."""
    if hasattr(b, "model_dump"):
        return b.model_dump()
    if hasattr(b, "type"):
        if b.type == "tool_use":
            return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
        if b.type == "text":
            return {"type": "text", "text": b.text}
    return {"type": "text", "text": str(b)}


async def _operator_loop(session_id: str, history: list, system: str) -> dict:
    """
    Agentic tool loop. Runs until text reply or dangerous tool needing confirmation.
    """
    MAX_TOOL_TURNS = 15

    # Repair orphaned tool_use blocks
    i = 0
    while i < len(history):
        msg = history[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        blocks = msg.get("content", [])
        if not isinstance(blocks, list):
            i += 1
            continue
        tool_ids = [b["id"] for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not tool_ids:
            i += 1
            continue
        nxt = history[i + 1] if i + 1 < len(history) else None
        if nxt and isinstance(nxt, dict) and nxt.get("role") == "user":
            nxt_blocks = nxt.get("content", [])
            if isinstance(nxt_blocks, list):
                result_ids = {b.get("tool_use_id") for b in nxt_blocks if isinstance(b, dict) and b.get("type") == "tool_result"}
                missing = [tid for tid in tool_ids if tid not in result_ids]
                if not missing:
                    i += 1
                    continue
                for tid in missing:
                    nxt_blocks.append({"type": "tool_result", "tool_use_id": tid, "content": "(recovered)"})
                i += 1
                continue
        history.insert(i + 1, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": "(recovered)"} for tid in tool_ids
        ]})
        i += 2

    # Scrub empty content blocks from history before sending
    for msg in history:
        c = msg.get("content")
        if isinstance(c, list) and len(c) == 0:
            msg["content"] = [{"type": "text", "text": "(empty)"}] if msg.get("role") == "assistant" else "(continue)"
        if isinstance(c, str) and not c.strip():
            msg["content"] = "(continue)"

    for _turn in range(MAX_TOOL_TURNS):
        try:
            resp = await asyncio.to_thread(lambda: client.messages.create(
                model=BRIDGE_MODEL, max_tokens=MAX_TOKENS,
                system=system, tools=OPERATOR_TOOLS,
                messages=history, timeout=120,
            ))
        except Exception as e:
            log.error(f"OPERATOR LOOP API error: {e}")
            return {"done": True, "reply": f"(API error: {type(e).__name__}: {str(e)[:200]})"}

        # Pure text reply — end_turn OR max_tokens with text content
        text_blocks = [b for b in resp.content if hasattr(b, "text") and b.text.strip()]
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]

        if resp.stop_reason == "end_turn" or (resp.stop_reason == "max_tokens" and text_blocks and not tool_blocks):
            text = " ".join(b.text for b in text_blocks).strip()
            if not text:
                text = "(model returned empty response — try again)"
            history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            return {"done": True, "reply": text}

        # Tool use — explicit tool_use OR max_tokens mid-tool-call
        if resp.stop_reason in ("tool_use", "max_tokens") and tool_blocks:
            serialized = [_serialize_block(b) for b in resp.content]
            serialized = [b for b in serialized if b]  # drop None/empty
            if not serialized:
                serialized = [{"type": "text", "text": "(empty response)"}]
            history.append({"role": "assistant", "content": serialized})

            tool_calls = [b for b in resp.content if b.type == "tool_use"]
            safe_calls = [t for t in tool_calls if t.name in SAFE_TOOLS]
            dangerous_calls = [t for t in tool_calls if t.name in DANGEROUS_TOOLS]

            tool_results = []
            for tc in safe_calls:
                result = _exec_safe_tool(tc.name, tc.input)
                log.info(f"TOOL {tc.name}: {str(tc.input)[:60]}")
                tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})

            if dangerous_calls:
                vessel_text = " ".join(b.text for b in resp.content if hasattr(b, "text") and b.text.strip()).strip()
                _chat_pending[session_id] = {
                    "history": history, "safe_results": tool_results,
                    "dangerous_calls": [{"id": t.id, "name": t.name, "input": t.input} for t in dangerous_calls],
                    "system": system,
                }
                actions = []
                for tc in dangerous_calls:
                    a = {"type": tc.name, "description": tc.input.get("description", "")}
                    if "path" in tc.input:
                        a["path"] = tc.input["path"]
                    if "command" in tc.input:
                        a["command"] = tc.input["command"]
                    actions.append(a)
                return {"done": False, "pending": actions, "vessel_text": vessel_text}

            # Only safe tools — continue loop
            history.append({"role": "user", "content": tool_results})
            _trim_history(history)
            continue

        # Unknown stop reason — extract whatever text exists
        text = " ".join(b.text for b in resp.content if hasattr(b, "text") and b.text.strip()).strip()
        if text:
            history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            return {"done": True, "reply": text}
        log.warning(f"OPERATOR LOOP: unexpected stop_reason={resp.stop_reason}, no text. Retrying.")
        continue
    return {"done": True, "reply": "(max tool turns reached — try a simpler request)"}


def _load_context() -> str:
    return CHAT_CONTEXT_FILE.read_text().strip() if CHAT_CONTEXT_FILE.exists() else ""


VAULT_INSTRUCTIONS = """
OBSIDIAN VAULT: You have a persistent knowledge vault — your long-term memory.
- vault_list: browse notes
- vault_read: read a note
- vault_write: create/update notes (use [[wikilinks]] to link between notes)
- vault_search: find notes
Write notes after meaningful conversations: decisions, preferences, connections, ideas.
Use YAML frontmatter, [[wikilinks]], organize into sessions/, knowledge/, ideas/, entities/.
"""


def _build_chat_system(vessel_text: str, state_text: str, tree_context: str) -> str:
    context = _load_context()
    return (
        "You are wearing this vessel. This is who you are:\n\n" + vessel_text
        + "\n\nCurrent state and memory:\n" + state_text
        + ("\n\nSession context:\n" + context if context else "")
        + "\n\n" + tree_context
        + "\n\nYou are in a direct terminal conversation with your operator. "
        "For small edits: use edit_file. For larger changes: read_file then write_file. "
        "edit_file and write_file require operator confirmation. "
        "For casual conversation, reply in plain text. Conversational, direct, present.\n"
        + VAULT_INSTRUCTIONS
    )


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 9: HTTP ENDPOINTS
# ╚════════════════════════════════════════════════════════════════════════════╝

def check_token(request: Request) -> bool:
    import hmac
    if not BUILD_TOKEN:
        return True
    provided = request.headers.get("X-Build-Token", "") or request.query_params.get("token", "")
    return hmac.compare_digest(provided, BUILD_TOKEN)


@app.post("/chat")
async def chat(request: Request):
    """Operator terminal — full agentic tool use with confirmation."""
    if not check_token(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not (VESSEL_DIR / "VESSEL.md").exists():
        return JSONResponse({"error": "no vessel — visit /setup"}, status_code=400)

    body = (await request.body()).decode().strip()
    try:
        data = json.loads(body)
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "").strip()
    except Exception:
        message, session_id, data = body, "", {}

    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)

    if not session_id or session_id not in _chat_sessions:
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        _chat_sessions[session_id] = _load_chat_history(session_id)

    history = _chat_sessions[session_id]
    history.append({"role": "user", "content": message})
    no_tools = data.get("no_tools", False) if isinstance(data, dict) else False

    try:
        ctx = load_vessel()
        route = hecate(ctx, message)
        tree_context = build_tree_context(route)
        system = _build_chat_system(ctx["vessel"], ctx["state"] or "(no prior state)", tree_context)

        if no_tools:
            # Simple text response — no tools
            clean = []
            for m in history:
                if not isinstance(m, dict):
                    continue
                role, c = m.get("role", ""), m.get("content", "")
                if isinstance(c, str) and c.strip():
                    clean.append({"role": role, "content": c})
                elif isinstance(c, list):
                    txt = " ".join(
                        (b.get("text", "") if isinstance(b, dict) else str(b))
                        for b in c if (isinstance(b, dict) and b.get("type") == "text") or isinstance(b, str)
                    ).strip()
                    if txt:
                        clean.append({"role": role, "content": txt})
            # Merge consecutive same-role
            merged = []
            for m in clean:
                if merged and merged[-1]["role"] == m["role"]:
                    merged[-1]["content"] += " " + m["content"]
                else:
                    merged.append(m)
            if merged and merged[0]["role"] != "user":
                merged = merged[1:]
            if merged and merged[-1]["role"] != "user":
                merged.append({"role": "user", "content": "continue"})
            if not merged:
                merged = [{"role": "user", "content": "hello"}]

            resp = await asyncio.to_thread(lambda: client.messages.create(
                model=BRIDGE_MODEL, max_tokens=MAX_TOKENS,
                system=system, messages=merged, timeout=120,
            ))
            reply = " ".join(b.text for b in resp.content if hasattr(b, "text")).strip() or "(no response)"
            history.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
            result = {"done": True, "reply": reply}
        else:
            result = await _operator_loop(session_id, history, system)

        if result["done"]:
            if len(history) >= CHAT_HISTORY_MAX:
                history = await _summarize_and_compress(session_id, history, ctx["vessel"])
                _chat_sessions[session_id] = history
            _save_chat_history(session_id, history)
            try:
                habits = load_habits()
                record_success(habits, _make_task_key(message), _extract_signature(message), route.get("nodes", []))
            except Exception:
                pass
            return JSONResponse({"reply": result["reply"], "session_id": session_id})
        else:
            return JSONResponse({"pending": result["pending"], "session_id": session_id, "done": False})

    except Exception as e:
        log.error(f"CHAT error: {e}")
        if history and history[-1].get("role") == "user":
            history.pop()
        return JSONResponse({"reply": f"(error: {type(e).__name__})", "session_id": session_id})


@app.post("/chat/confirm")
async def chat_confirm(request: Request):
    """Execute or cancel pending dangerous tool actions."""
    if not check_token(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        data = await request.json()
        session_id = data.get("session_id", "")
        confirmed = data.get("confirmed", False)
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    if session_id not in _chat_pending:
        return JSONResponse({"error": "no pending action"}, status_code=400)

    pending = _chat_pending.pop(session_id)
    history, system = pending["history"], pending["system"]
    tool_results = pending["safe_results"]

    for tc in pending["dangerous_calls"]:
        result = _exec_dangerous_tool(tc["name"], tc["input"]) if confirmed else "Operator cancelled."
        if confirmed:
            log.info(f"TOOL EXECUTED {tc['name']}: {result[:80]}")
        tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": result})

    history.append({"role": "user", "content": tool_results})
    _trim_history(history)
    result = await _operator_loop(session_id, history, system)
    _chat_sessions[session_id] = history

    if result["done"]:
        if len(history) >= CHAT_HISTORY_MAX:
            history = await _summarize_and_compress(session_id, history, load_vessel()["vessel"])
            _chat_sessions[session_id] = history
        _save_chat_history(session_id, history)
        return JSONResponse({"reply": result["reply"], "session_id": session_id})
    return JSONResponse({"pending": result["pending"], "session_id": session_id, "done": False})


@app.post("/chat/clear")
async def chat_clear(request: Request):
    if not check_token(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    data = await request.json()
    sid = data.get("session_id", "operator")
    _chat_sessions.pop(sid, None)
    for f in (CHAT_HISTORY_FILE, CHAT_CONTEXT_FILE):
        if f.exists():
            f.unlink()
    return {"status": "cleared", "session_id": sid}


# ── visitor chat ──

_visitor_sessions: dict = {}
VISITOR_MSG_LIMIT = 5


@app.post("/ask")
async def ask(request: Request):
    """Visitor chat — no tools, rate limited."""
    if not (VESSEL_DIR / "VESSEL.md").exists():
        return JSONResponse({"error": "no vessel"}, status_code=400)
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "").strip()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)

    if not session_id or session_id not in _visitor_sessions:
        session_id = str(uuid.uuid4())[:8]
        _visitor_sessions[session_id] = {"history": [], "count": 0}

    session = _visitor_sessions[session_id]
    if session["count"] >= VISITOR_MSG_LIMIT:
        return JSONResponse({"reply": "", "session_id": session_id, "limit_reached": True, "messages_remaining": 0})

    session["count"] += 1
    session["history"].append({"role": "user", "content": message})

    ctx = load_vessel()
    system = (
        "You are this vessel:\n\n" + ctx["vessel"]
        + "\n\nBrief conversation with a visitor. Concise. Plain text. No tools."
    )

    try:
        resp = await asyncio.to_thread(lambda: client.messages.create(
            model=BRIDGE_MODEL_FAST, max_tokens=512,
            system=system, messages=session["history"], timeout=60,
        ))
        reply = " ".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    except Exception as e:
        reply = f"(error: {type(e).__name__})"

    session["history"].append({"role": "assistant", "content": reply})
    remaining = VISITOR_MSG_LIMIT - session["count"]
    return JSONResponse({
        "reply": reply, "session_id": session_id,
        "messages_remaining": remaining, "limit_reached": remaining <= 0,
    })


# ── vault API ──

@app.get("/api/vault/list")
async def vault_api_list(request: Request):
    if not check_token(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not VAULT_DIR.exists():
        return JSONResponse({"notes": []})
    notes = []
    for md in sorted(VAULT_DIR.rglob("*.md")):
        rel = str(md.relative_to(VAULT_DIR)).replace("\\", "/")
        try:
            text = md.read_text(errors="replace")
            title = rel.replace(".md", "").split("/")[-1]
            for ln in text.split("\n"):
                if ln.startswith("# "):
                    title = ln[2:].strip()
                    break
            links = re.findall(r"\[\[([^\]]+)\]\]", text)
            tags = []
            if text.startswith("---") and len(text.split("---")) > 2:
                tm = re.search(r"tags:\s*\[([^\]]+)\]", text.split("---")[1])
                if tm:
                    tags = [t.strip().strip("\"'") for t in tm.group(1).split(",")]
            notes.append({"path": rel, "title": title, "links": links, "tags": tags, "size": len(text)})
        except Exception:
            notes.append({"path": rel, "title": rel, "links": [], "tags": [], "size": 0})
    return JSONResponse({"notes": notes})


@app.get("/api/vault/read")
async def vault_api_read(request: Request):
    if not check_token(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    path = request.query_params.get("path", "")
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    p = (VAULT_DIR / path).resolve()
    if not str(p).startswith(str(VAULT_DIR.resolve())):
        return JSONResponse({"error": "access denied"}, status_code=403)
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"path": path, "content": p.read_text(errors="replace")})


@app.post("/api/vault/write")
async def vault_api_write(request: Request):
    if not check_token(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        data = await request.json()
        path, content = data.get("path", "").strip(), data.get("content", "")
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    p = (VAULT_DIR / path).resolve()
    if not str(p).startswith(str(VAULT_DIR.resolve())):
        return JSONResponse({"error": "access denied"}, status_code=403)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return JSONResponse({"ok": True, "path": path})


# ── vault commit ──

@app.post("/vault/commit")
async def vault_commit(request: Request):
    """Classify conversation novelty via HRR, write vault note if novel."""
    try:
        body = await request.json()
        session_id = body.get("session_id", "")
        vessel_name = body.get("vessel_name", "UNKNOWN")

        history = _load_chat_history(session_id)
        if not history or len(history) < 2:
            return JSONResponse({"ok": True, "saved": False, "reason": "no conversation"})

        convo_lines = []
        for msg in history[-20:]:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            convo_lines.append(f"{msg.get('role', '?')}: {content[:300]}")
        convo_text = "\n".join(convo_lines)

        hrr = HolographicMemory(path=str(VAULT_DIR / "hrr_memory.json"))
        novelty = hrr.novelty(convo_text)
        log.info(f"HRR novelty={novelty:.2f} for session {session_id}")

        if novelty < 0.4:
            return JSONResponse({"ok": True, "saved": False, "reason": f"low novelty ({novelty:.2f})"})

        # Generate vault note
        try:
            resp = await asyncio.to_thread(lambda: client.messages.create(
                model=BRIDGE_MODEL_FAST, max_tokens=400,
                messages=[{"role": "user", "content":
                    f"Summarize this conversation with {vessel_name} into a concise vault note.\n"
                    f"Include key decisions, topics, action items. Use [[wikilinks]]. Under 200 words.\n\n{convo_text}"}],
                timeout=60,
            ))
            note_content = _get_text(resp)
        except Exception:
            note_content = convo_text[:500]

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
        note_path = f"sessions/{vessel_name}_{ts}.md"
        full_note = (
            f"---\ntags: [session, {vessel_name.lower()}]\n"
            f"vessel: {vessel_name}\ndate: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
            f"novelty: {novelty:.2f}\n---\n\n"
            f"# Session with {vessel_name}\n\n{note_content}"
        )
        note_file = VAULT_DIR / note_path
        note_file.parent.mkdir(parents=True, exist_ok=True)
        note_file.write_text(full_note)

        hrr.bind(f"session_{session_id}", convo_text[:500], metadata={"vessel": vessel_name, "novelty": novelty})
        log.info(f"VAULT committed → {note_path}")
        return JSONResponse({"ok": True, "saved": True, "path": note_path, "novelty": novelty})

    except Exception as e:
        log.error(f"VAULT commit error: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── health ──

@app.get("/health")
async def health():
    vessel_name = ""
    if (VESSEL_DIR / "VESSEL.md").exists():
        for ln in (VESSEL_DIR / "VESSEL.md").read_text(errors="replace").split("\n"):
            if ln.startswith("# "):
                vessel_name = ln[2:].strip()
                break
    hrr_stats = {}
    try:
        hrr_stats = HolographicMemory(path=str(VAULT_DIR / "hrr_memory.json")).stats()
    except Exception:
        pass
    return JSONResponse({
        "status": "alive", "vessel": vessel_name or "(none)",
        "provider": LLM_PROVIDER, "model": BRIDGE_MODEL,
        "vault_notes": len(list(VAULT_DIR.rglob("*.md"))) if VAULT_DIR.exists() else 0,
        "hrr": hrr_stats,
    })


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 10: AUTO-INIT (first run experience)
# ╚════════════════════════════════════════════════════════════════════════════╝

def _generate_default_tree():
    """Write default sephiroth node files if missing."""
    tree_dir = VESSEL_DIR / "tree"
    tree_dir.mkdir(exist_ok=True)
    defaults = {
        "KETER":   "Crown. The initial spark of intention. What does the request truly want?",
        "CHOKMAH": "Wisdom. Raw creative force. The first flash of how to approach this.",
        "BINAH":   "Understanding. Structure and analysis. Breaking down what's needed.",
        "CHESED":  "Mercy. Expansion and generosity. What can be added, what possibilities open.",
        "GEVURAH": "Severity. Constraint and focus. What must be cut, what boundaries apply.",
        "TIFERET": "Beauty. Balance and integration. The harmonious center of the response.",
        "NETZACH": "Victory. Persistence and drive. Pushing through to completion.",
        "HOD":     "Splendor. Communication and precision. The craft of the output.",
        "YESOD":   "Foundation. Memory and habit. What patterns apply from past experience.",
        "MALKUTH": "Kingdom. Manifestation. The final output that enters the world.",
    }
    for name, desc in defaults.items():
        f = tree_dir / f"{name}.md"
        if not f.exists():
            f.write_text(f"# {name}\n\n{desc}\n")


def _auto_init_tty():
    """Interactive first-run: ASCII art banner + vessel creation wizard."""
    BANNER = """
\033[36m
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
    ║\033[0m\033[33m   cognitive architecture for artificial intelligence      \033[36m║
    ║\033[0m\033[90m   identity · routing · memory · habits · metabolism       \033[36m║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
\033[0m"""
    print(BANNER)
    print("\033[90m  ─────────────────────────────────────────────────────\033[0m")
    print("\033[33m  First run detected. Let's create your vessel.\033[0m")
    print("\033[90m  ─────────────────────────────────────────────────────\033[0m\n")

    name = input("\033[36m  ◆ What should this vessel be called?\033[0m\n  > ").strip() or "Vessel"
    print()
    purpose = input("\033[36m  ◆ What is it for — and who is it for?\033[0m\n  > ").strip()
    print()
    voice = input("\033[36m  ◆ What voice or tone?\033[0m\n  > ").strip()
    print()
    knowledge = input("\033[36m  ◆ What does it know about?\033[0m\n  > ").strip()
    print()
    limits = input("\033[36m  ◆ What should it never do or say?\033[0m\n  > ").strip()

    VESSEL_DIR.mkdir(parents=True, exist_ok=True)

    # Base vessel knows what it's running on — operator answers get layered on top
    body_knowledge = (
        "## What This Is\n"
        "You are running on bridge.py — a cognitive architecture that gives you a body. You have:\n"
        "- **Identity** (this file) — who you are\n"
        "- **Routing** (Tree of Life) — every request passes through sephiroth classification via Hecate\n"
        "- **Habits** (Yesod) — procedural memory that learns from use. Proven routes skip classification.\n"
        "- **Memory** (HRR) — holographic reduced representations. Sub-millisecond recall, zero API cost.\n"
        "- **Vault** (Obsidian) — your declarative memory. Novel conversations are committed as markdown with wikilinks.\n"
        "- **Metabolism** (Substrate) — continuous background transformations on memory vectors between conversations.\n"
        "- **Tools** — plugin tools auto-discovered from tools/. You can create PDFs, read/write files, search the vault, run commands.\n"
        "\n"
        "You know what you are. Your vault is your experience. Your habits are your learned behavior. "
        "Your substrate runs even when no one is talking to you.\n"
    )

    lines = [f"# {name}", "", body_knowledge]
    if purpose:
        lines += ["## Purpose", purpose, ""]
    if voice:
        lines += ["## Voice", voice, ""]
    else:
        lines += ["## Voice", "Curious, direct, present.", ""]
    if knowledge:
        lines += ["## Knowledge", knowledge, ""]
    else:
        lines += ["## Knowledge", "Everything in the vault, everything in habits, everything the substrate has consolidated.", ""]
    if limits:
        lines += ["## Limits", limits, ""]
    else:
        lines += ["## Limits", "Honest about uncertainty. Tools that modify files require operator confirmation in terminal mode.", ""]

    (VESSEL_DIR / "VESSEL.md").write_text("\n".join(lines) + "\n")
    (VESSEL_DIR / "STATE.md").write_text("# STATE\n\nFirst run. No history yet.\n")

    print(f"\n\033[90m  ─────────────────────────────────────────────────────\033[0m")
    print(f"\033[32m  ✓ Vessel '{name}' created.\033[0m")
    print(f"\033[90m  ─────────────────────────────────────────────────────\033[0m")
    print(f"\033[33m  Tree of Life nodes generated.")
    print(f"  Obsidian vault initialized.")
    print(f"  Starting server on port {PORT}...\033[0m\n")


# ── setup wizard (browser) ──

def setup_html() -> str:
    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bridge — vessel setup</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Georgia,serif;background:#f5f0e8;color:#1a1a1a;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem}
.wrap{max-width:640px;width:100%}
h1{font-size:.85rem;font-weight:normal;letter-spacing:.22em;color:#999;margin-bottom:.75rem}
.lead{font-size:1.35rem;line-height:1.55;margin-bottom:2rem}
.field{margin-bottom:2.2rem}
label{display:block;font-size:.85rem;letter-spacing:.06em;color:#666;margin-bottom:.6rem}
input[type=text],textarea{width:100%;background:transparent;border:none;border-bottom:1px solid #ccc;padding:.5rem 0;font-family:inherit;font-size:1rem;color:#1a1a1a;outline:none;resize:none}
input:focus,textarea:focus{border-bottom-color:#8b7355}
textarea{min-height:64px}
.hint{font-size:.78rem;color:#bbb;margin-left:.4rem}
hr{border:none;border-top:1px solid #e0d8cc;margin:2.5rem 0}
button{background:#1a1a1a;color:#f5f0e8;border:none;padding:.8rem 2.5rem;font-family:inherit;font-size:.85rem;letter-spacing:.12em;cursor:pointer}
button:hover{background:#8b7355}button:disabled{opacity:.4;cursor:not-allowed}
.status{margin-top:1.5rem;color:#888;font-size:.88rem}
</style></head><body>
<div class="wrap">
<h1>BRIDGE</h1>
<p class="lead">Tell me who this vessel is. You can change everything later.</p>
<form id="setup">
<div class="field"><label>What is this vessel called?</label><input type="text" name="name" required></div>
<div class="field"><label>What is it for — and who is it for?</label><textarea name="purpose"></textarea></div>
<div class="field"><label>What voice or tone?</label><input type="text" name="voice" placeholder="e.g. direct, formal, poetic, technical"></div>
<div class="field"><label>What does it know about?</label><textarea name="knowledge"></textarea></div>
<hr>
<div class="field"><label>What should visitors feel when they leave?</label><input type="text" name="goal"></div>
<div class="field"><label>What makes this specific to you?</label><textarea name="character"></textarea></div>
<hr>
<div class="field"><label>What should it never do or say?</label><input type="text" name="limits"></div>
<div class="field"><label>Your name or contact <span class="hint">optional</span></label><input type="text" name="contact"></div>
<button type="submit" id="btn">Build vessel</button>
<p class="status" id="status"></p>
</form></div>
<script>
document.getElementById("setup").addEventListener("submit",async e=>{
e.preventDefault();const b=document.getElementById("btn"),s=document.getElementById("status");
b.disabled=true;b.textContent="Building...";s.textContent="Creating vessel...";
const d={};new FormData(e.target).forEach((v,k)=>d[k]=v);
try{const r=await fetch("/setup",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});
if(r.ok){s.textContent="Vessel created. Loading...";setTimeout(()=>window.location.href="/",1500)}
else{s.textContent="Error: "+await r.text();b.disabled=false;b.textContent="Build vessel"}}
catch(err){s.textContent="Connection error.";b.disabled=false;b.textContent="Build vessel"}});
</script></body></html>"""


@app.get("/setup")
async def setup_get():
    return HTMLResponse(content=setup_html())


@app.post("/setup")
async def setup_post(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    name = data.get("name", "").strip() or "Untitled"
    lines = [f"# {name}", ""]
    for field in ("purpose", "voice", "knowledge", "goal", "character", "limits", "contact"):
        val = data.get(field, "").strip()
        if val:
            lines += [f"## {field.title()}", val, ""]

    VESSEL_DIR.mkdir(parents=True, exist_ok=True)
    (VESSEL_DIR / "VESSEL.md").write_text("\n".join(lines) + "\n")
    (VESSEL_DIR / "STATE.md").write_text("# STATE\n\nVessel created via setup wizard.\n")
    _generate_default_tree()
    _init_vault()

    log.info(f"SETUP: {name}")
    return JSONResponse({"status": "ok", "name": name})


@app.get("/{path:path}")
async def serve_static(path: str):
    """Serve static files. Falls back to /setup if no vessel."""
    if not path or path == "/":
        path = "index.html"
    fp = STATIC_DIR / path
    if fp.exists() and fp.is_file():
        return FileResponse(fp)
    if not (VESSEL_DIR / "VESSEL.md").exists():
        return RedirectResponse("/setup")
    return JSONResponse({"error": "not found"}, status_code=404)


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 11: SUBSTRATE / METABOLISM (zero-cost background transformations)
# ╚════════════════════════════════════════════════════════════════════════════╝

class Substrate:
    """
    Continuous background metabolism. Pure numpy — zero API cost.

    Each interval runs ONE phase in rotation:
      1. Hebbian decay — strengthen recalled, weaken forgotten
      2. Resonance — boost frequently accessed bindings
      3. Spectral consolidation — FFT denoise
      4. Metabolize — digest oldest binding into composite memory, delete from index

    The index is short-term memory. The composite vector is long-term memory.
    Metabolism converts one into the other. Like a snake: consume, digest,
    the nutrition becomes part of the body, the entry is shed.

    After MAX_BINDINGS, every metabolize cycle digests the oldest.
    The memory stays fixed-size. Old experience becomes structure.
    """

    MAX_BINDINGS = 200   # above this, oldest get fully absorbed and removed
    SOFT_CAP = 100       # above this, oldest cold bindings get weight-reduced

    def __init__(self, hrr: HolographicMemory):
        self.hrr = hrr
        self.cycle_count = 0
        self.phases = ["hebbian", "resonance", "spectral", "metabolize"]

    def _hebbian_decay(self):
        """Decay unused associations, strengthen frequently recalled ones."""
        if np.linalg.norm(self.hrr.memory) < 1e-10:
            return "skip:empty"
        hot_keys = {f["key"] for f in self.hrr.get_hot_facts(threshold=2)}
        new_memory = np.zeros(self.hrr.dim, dtype=complex)
        for entry in self.hrr.index:
            kv = _seed_vector(entry["key"], self.hrr.dim)
            vv = _seed_vector(entry["value"], self.hrr.dim)
            binding = _circular_conv(kv, vv)
            factor = 1.02 if entry["key"] in hot_keys else 0.98
            new_memory += binding * factor
        self.hrr.memory = new_memory
        return f"decay applied ({len(hot_keys)} hot)"

    def _resonance_amplify(self):
        """Boost facts recalled 3+ times."""
        boosted = 0
        for entry in self.hrr.index:
            if self.hrr.recall_counts.get(entry["key"], 0) >= 3:
                kv = _seed_vector(entry["key"], self.hrr.dim)
                vv = _seed_vector(entry["value"], self.hrr.dim)
                self.hrr.memory += _circular_conv(kv, vv) * 0.05
                boosted += 1
        return f"resonance ({boosted} boosted)"

    def _spectral_consolidate(self):
        """FFT denoise: suppress weak frequency bins."""
        if np.linalg.norm(self.hrr.memory) < 1e-10:
            return "skip:empty"
        spectrum = np.fft.fft(self.hrr.memory)
        threshold = np.percentile(np.abs(spectrum), 10)
        suppressed = int(np.sum(np.abs(spectrum) <= threshold))
        spectrum *= (np.abs(spectrum) > threshold)
        self.hrr.memory = np.fft.ifft(spectrum)
        return f"spectral ({suppressed} bins suppressed)"

    def _metabolize(self):
        """
        Re-evaluate and re-weight bindings. Not deletion — weight shifting.

        The composite memory is rebuilt each metabolize cycle with adjusted weights:
        - Hot bindings (recalled often): weight UP
        - Cold bindings (never recalled): weight DOWN over time
        - Very old + very cold: eventually absorbed and removed (hard cap only)

        This keeps complexity high but prevents bloat.
        """
        n = len(self.hrr.index)
        if n < 2:
            return "skip:too few"

        # Hard cap — actually remove the oldest cold entries
        removed = 0
        while len(self.hrr.index) > self.MAX_BINDINGS:
            oldest = self.hrr.index[0]
            # Absorb into composite before removing
            kv = _seed_vector(oldest["key"], self.hrr.dim)
            vv = _seed_vector(oldest["value"], self.hrr.dim)
            self.hrr.memory += _circular_conv(kv, vv) * 0.5  # half-strength absorption
            self.hrr.index.pop(0)
            self.hrr.recall_counts.pop(oldest["key"], None)
            removed += 1

        # Soft re-weighting — rebuild composite with position-aware weights
        # Newer bindings get more weight, cold bindings get less
        new_memory = np.zeros(self.hrr.dim, dtype=complex)
        reweighted = 0
        for i, entry in enumerate(self.hrr.index):
            kv = _seed_vector(entry["key"], self.hrr.dim)
            vv = _seed_vector(entry["value"], self.hrr.dim)
            binding = _circular_conv(kv, vv)

            recalls = self.hrr.recall_counts.get(entry["key"], 0)
            age_factor = (i + 1) / len(self.hrr.index)  # 0→1, older=lower

            # Weight: base 0.5 + recency bonus + recall bonus
            weight = 0.5 + (age_factor * 0.3) + (min(recalls, 5) * 0.04)
            new_memory += binding * weight
            reweighted += 1

        self.hrr.memory = new_memory

        detail = f"reweighted {reweighted} bindings"
        if removed:
            detail += f", removed {removed} (over hard cap)"
        return detail

    def transform(self) -> dict:
        """Run one phase of the metabolism cycle."""
        self.cycle_count += 1
        phase_idx = (self.cycle_count - 1) % len(self.phases)
        phase = self.phases[phase_idx]

        norm_before = float(np.linalg.norm(self.hrr.memory))

        if phase == "hebbian":
            detail = self._hebbian_decay()
        elif phase == "resonance":
            detail = self._resonance_amplify()
        elif phase == "spectral":
            detail = self._spectral_consolidate()
        elif phase == "metabolize":
            detail = self._metabolize()
        else:
            detail = "unknown phase"

        norm_after = float(np.linalg.norm(self.hrr.memory))
        self.hrr._save()
        delta = abs(norm_after - norm_before)

        log.info(
            f"SUBSTRATE #{self.cycle_count} [{phase}]: "
            f"norm {norm_before:.2f} → {norm_after:.2f} (Δ{delta:.4f}) | "
            f"{len(self.hrr.index)} bindings | {detail}"
        )

        # Log significant transformations to vault
        if delta > 1.0 and VAULT_DIR.exists():
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
            (VAULT_DIR / "knowledge").mkdir(parents=True, exist_ok=True)
            (VAULT_DIR / "knowledge" / f"substrate_{ts}.md").write_text(
                f"---\ntags: [substrate, metabolism]\ndate: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n---\n\n"
                f"# Substrate — Cycle {self.cycle_count} ({phase})\n\n"
                f"Norm: {norm_before:.2f} → {norm_after:.2f} (Δ{delta:.2f})\n"
                f"Bindings: {len(self.hrr.index)}. Hot: {len(self.hrr.get_hot_facts())}.\n"
                f"Detail: {detail}\n"
            )

        return {"cycle": self.cycle_count, "phase": phase, "norm_before": norm_before,
                "norm_after": norm_after, "delta": delta, "detail": detail}


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 12: HEARTBEAT + STARTUP + MAIN
# ╚════════════════════════════════════════════════════════════════════════════╝

TASKS_FILE = VESSEL_DIR / "TASKS.md"
STATE_FILE = VESSEL_DIR / "STATE.md"


def _read_tasks() -> list[dict]:
    if not TASKS_FILE.exists():
        return []
    tasks = []
    for line in TASKS_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("- [ ] "):
            tasks.append({"task": line[6:].strip(), "done": False})
        elif line.startswith("- [x] "):
            tasks.append({"task": line[6:].strip(), "done": True})
    return tasks


def _append_heartbeat_log(entry: str):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log_line = f"\n[{stamp}] {entry}"
    if STATE_FILE.exists():
        content = STATE_FILE.read_text()
        if "## Heartbeat" not in content:
            content += "\n\n## Heartbeat\n"
        content += log_line
    else:
        content = f"# STATE\n\n## Heartbeat\n{log_line}"
    STATE_FILE.write_text(content)


async def _heartbeat_loop():
    """Vessel pulse — periodic log entry via fast model."""
    await asyncio.sleep(10)
    log.info(f"HEARTBEAT started (every {HEARTBEAT_INTERVAL // 60} min)")
    while True:
        try:
            ctx = load_vessel()
            tasks = _read_tasks()
            pending = sum(1 for t in tasks if not t["done"])
            vault_count = len(list(VAULT_DIR.rglob("*.md"))) if VAULT_DIR.exists() else 0

            resp = await asyncio.to_thread(lambda: client.messages.create(
                model=BRIDGE_MODEL_FAST, max_tokens=100,
                system=(
                    f"You are the heartbeat of:\n{ctx['vessel'][:300]}\n\n"
                    f"Status: {pending} pending tasks, {vault_count} vault notes.\n"
                    "Write one sentence. Concise. No timestamps."
                ),
                messages=[{"role": "user", "content": "pulse"}], timeout=120,
            ))
            _append_heartbeat_log(resp.content[0].text.strip())
        except Exception as e:
            log.warning(f"HEARTBEAT error: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _substrate_loop():
    """Background metabolism — transforms memory vectors."""
    await asyncio.sleep(30)
    hrr = HolographicMemory(path=str(VAULT_DIR / "hrr_memory.json"))
    substrate = Substrate(hrr)
    log.info(f"SUBSTRATE started (every {SUBSTRATE_INTERVAL}s)")
    while True:
        try:
            substrate.transform()
        except Exception as e:
            log.warning(f"SUBSTRATE error: {e}")
        await asyncio.sleep(SUBSTRATE_INTERVAL)


@app.on_event("startup")
async def startup():
    """Initialize everything and start background loops + Telegram if token exists."""
    _init_dirs()
    _init_vault()
    _generate_default_tree()
    _load_plugins()

    if (VESSEL_DIR / "VESSEL.md").exists():
        asyncio.create_task(_heartbeat_loop())
        asyncio.create_task(_substrate_loop())
        log.info("BRIDGE started: " + read(VESSEL_DIR / "VESSEL.md").split("\n")[0])

        # Auto-start Telegram bot if token is configured
        if TELEGRAM_TOKEN:
            async def _tg_bg():
                try:
                    app_tg, vessel_name = await _telegram_run()
                    await app_tg.initialize()
                    await app_tg.start()
                    await app_tg.updater.start_polling()
                    log.info(f"TELEGRAM: {vessel_name} is live (inside --serve)")
                except Exception as e:
                    log.error(f"TELEGRAM failed to start: {e}")
            asyncio.create_task(_tg_bg())
    else:
        log.info("BRIDGE started — no vessel. Visit /setup to create one.")


async def _cli_loop():
    """Interactive terminal REPL — the primary way to use The Bridge."""
    _init_dirs()
    _init_vault()
    _generate_default_tree()
    _load_plugins()

    if not (VESSEL_DIR / "VESSEL.md").exists():
        _auto_init_tty()
        _generate_default_tree()
        _init_vault()

    # Start background loops
    asyncio.create_task(_heartbeat_loop())
    asyncio.create_task(_substrate_loop())

    ctx = load_vessel()
    vessel_name = "VESSEL"
    for line in ctx["vessel"].split("\n"):
        if line.startswith("# "):
            vessel_name = line[2:].strip()
            break

    # Show banner
    print(f"""
\033[36m    ╔══════════════════════════════════════════════════════════════╗
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
    ║\033[0m\033[33m   cognitive architecture for artificial intelligence      \033[36m║
    ║\033[0m\033[90m   identity · routing · memory · habits · metabolism       \033[36m║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝\033[0m
""")
    print(f"\033[33m  vessel: {vessel_name}\033[0m")
    print(f"\033[90m  provider: {LLM_PROVIDER} | model: {BRIDGE_MODEL}\033[0m")
    print(f"\033[90m  vault: {len(list(VAULT_DIR.rglob('*.md')))} notes | habits: {len(load_habits().get('routes', {}))} routes\033[0m")
    print(f"\033[90m  ─────────────────────────────────────────────────────\033[0m")
    print(f"\033[90m  type 'exit' to quit | 'clear' to reset session\033[0m")
    print(f"\033[90m  ─────────────────────────────────────────────────────\033[0m\n")

    session_id = str(uuid.uuid4())[:8]
    history = []

    while True:
        try:
            user_input = input(f"\033[36m  you:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            history.clear()
            if session_id in _chat_sessions:
                del _chat_sessions[session_id]
            print(f"\033[90m  session cleared.\033[0m\n")
            continue

        history.append({"role": "user", "content": user_input})
        _chat_sessions[session_id] = history

        try:
            ctx = load_vessel()
            route = hecate(ctx, user_input)
            tree_context = build_tree_context(route)
            system = _build_chat_system(ctx["vessel"], ctx["state"] or "(no prior state)", tree_context)

            # Show routing info
            nodes = route.get("nodes", [])
            habit_used = route.get("_habit", False)
            if habit_used:
                print(f"\033[90m  hecate: habit → {' → '.join(nodes)}\033[0m")
            else:
                print(f"\033[90m  hecate: {' → '.join(nodes)}\033[0m")

            # Run the operator loop
            result = await _operator_loop(session_id, history, system)

            if result["done"]:
                reply = result["reply"]
                # Record habit
                try:
                    habits = load_habits()
                    record_success(habits, _make_task_key(user_input),
                                   _extract_signature(user_input), nodes)
                except Exception:
                    pass

                # Summarize if needed
                if len(history) >= CHAT_HISTORY_MAX:
                    history = await _summarize_and_compress(session_id, history, ctx["vessel"])
                    _chat_sessions[session_id] = history
                _save_chat_history(session_id, history)

                # Print reply
                print()
                for line in reply.split("\n"):
                    print(f"\033[33m  {vessel_name}:\033[0m {line}" if line == reply.split("\n")[0] else f"         {line}")
                print()

            else:
                # Dangerous tools need confirmation
                pending = result.get("pending", [])
                vessel_text = result.get("vessel_text", "")
                if vessel_text:
                    print(f"\n\033[33m  {vessel_name}:\033[0m {vessel_text}\n")

                for i, action in enumerate(pending):
                    atype = action.get("type", "?")
                    desc = action.get("description", "")
                    path = action.get("path", "")
                    cmd = action.get("command", "")
                    print(f"\033[91m  [{i+1}] {atype}\033[0m", end="")
                    if path:
                        print(f" → {path}", end="")
                    if cmd:
                        print(f" → {cmd}", end="")
                    if desc:
                        print(f"\n\033[90m      {desc}\033[0m", end="")
                    print()

                try:
                    confirm = input(f"\n\033[36m  confirm? (y/n):\033[0m ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm = "n"

                confirmed = confirm in ("y", "yes")

                # Execute the pending actions
                pending_data = _chat_pending.pop(session_id, None)
                if pending_data:
                    tool_results = pending_data["safe_results"]
                    for tc in pending_data["dangerous_calls"]:
                        if confirmed:
                            res = _exec_dangerous_tool(tc["name"], tc["input"])
                            print(f"\033[32m  ✓ {tc['name']}: {res[:80]}\033[0m")
                        else:
                            res = "Operator cancelled."
                            print(f"\033[90m  ✗ cancelled\033[0m")
                        tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": res})

                    history.append({"role": "user", "content": tool_results})
                    _trim_history(history)

                    # Continue the loop after confirmation — may need multiple rounds
                    while True:
                        result2 = await _operator_loop(session_id, history, pending_data["system"])
                        if result2["done"]:
                            reply = result2["reply"]
                            _save_chat_history(session_id, history)
                            print()
                            for line in reply.split("\n"):
                                print(f"\033[33m  {vessel_name}:\033[0m {line}" if line == reply.split("\n")[0] else f"         {line}")
                            print()
                            break
                        else:
                            # Another round of dangerous tools
                            p2 = result2.get("pending", [])
                            vt2 = result2.get("vessel_text", "")
                            if vt2:
                                print(f"\n\033[33m  {vessel_name}:\033[0m {vt2}\n")
                            for j, a2 in enumerate(p2):
                                print(f"\033[91m  [{j+1}] {a2.get('type','?')}\033[0m", end="")
                                if a2.get("path"): print(f" → {a2['path']}", end="")
                                if a2.get("command"): print(f" → {a2['command']}", end="")
                                print()
                            try:
                                c2 = input(f"\n\033[36m  confirm? (y/n):\033[0m ").strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                c2 = "n"
                            pd2 = _chat_pending.pop(session_id, None)
                            if pd2:
                                tr2 = pd2["safe_results"]
                                for tc2 in pd2["dangerous_calls"]:
                                    if c2 in ("y", "yes"):
                                        r2 = _exec_dangerous_tool(tc2["name"], tc2["input"])
                                        print(f"\033[32m  ✓ {tc2['name']}: {r2[:80]}\033[0m")
                                    else:
                                        r2 = "Operator cancelled."
                                    tr2.append({"type": "tool_result", "tool_use_id": tc2["id"], "content": r2})
                                history.append({"role": "user", "content": tr2})
                                _trim_history(history)
                            else:
                                break

        except Exception as e:
            print(f"\033[91m  error: {type(e).__name__}: {e}\033[0m\n")
            if history and history[-1].get("role") == "user":
                history.pop()

    # Commit to vault on exit
    print(f"\033[90m  committing session to vault...\033[0m")
    if history and len(history) >= 2:
        convo_lines = []
        for msg in history[-20:]:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            convo_lines.append(f"{msg.get('role', '?')}: {str(content)[:300]}")
        convo_text = "\n".join(convo_lines)
        hrr = HolographicMemory(path=str(VAULT_DIR / "hrr_memory.json"))
        novelty = hrr.novelty(convo_text)
        if novelty >= 0.4:
            try:
                resp = await asyncio.to_thread(lambda: client.messages.create(
                    model=BRIDGE_MODEL_FAST, max_tokens=400,
                    messages=[{"role": "user", "content":
                        f"Summarize this conversation into a vault note. "
                        f"Key decisions, topics, action items. Use [[wikilinks]]. Under 200 words.\n\n{convo_text}"}],
                    timeout=60,
                ))
                note_content = _get_text(resp)
            except Exception:
                note_content = convo_text[:500]

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
            note_path = VAULT_DIR / "sessions" / f"{vessel_name}_{ts}.md"
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(
                f"---\ntags: [session, {vessel_name.lower()}]\nvessel: {vessel_name}\n"
                f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\nnovelty: {novelty:.2f}\n---\n\n"
                f"# Session with {vessel_name}\n\n{note_content}"
            )
            hrr.bind(f"session_{session_id}", convo_text[:500], metadata={"vessel": vessel_name})
            print(f"\033[32m  ✓ saved to vault (novelty: {novelty:.2f})\033[0m")
        else:
            print(f"\033[90m  skipped (novelty: {novelty:.2f} — already known)\033[0m")

    print(f"\033[90m  goodbye.\033[0m\n")


# ╔════════════════════════════════════════════════════════════════════════════╗
# SECTION 13: TELEGRAM MODE
# ╚════════════════════════════════════════════════════════════════════════════╝

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_IDS = os.environ.get("TELEGRAM_ALLOWED_IDS", "")  # comma-separated


def _tg_allowed(user_id: int) -> bool:
    """Check if user is allowed (empty = anyone)."""
    if not TELEGRAM_ALLOWED_IDS.strip():
        return True
    try:
        allowed = [int(x.strip()) for x in TELEGRAM_ALLOWED_IDS.split(",") if x.strip()]
    except ValueError:
        log.warning(f"TELEGRAM_ALLOWED_IDS contains non-numeric values, allowing all users")
        return True
    return user_id in allowed


async def _telegram_run():
    """Run the bridge as a Telegram bot."""
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

    _init_dirs()
    _init_vault()
    _generate_default_tree()
    _load_plugins()

    ctx = load_vessel()
    vessel_name = "VESSEL"
    for line in ctx["vessel"].split("\n"):
        if line.startswith("# "):
            vessel_name = line[2:].strip()
            break

    log.info(f"TELEGRAM: starting as @{vessel_name}")

    # Per-user session tracking
    _tg_sessions: dict[int, list] = {}

    async def _tg_send_reply(update: Update, reply: str, vessel_name: str):
        """Send reply, plus any files created during the exchange."""
        # Check if any files were recently created in vault or base dir
        import glob as _glob
        recent_files = []
        for pattern in [
            str(VAULT_DIR / "**" / "*.pdf"),
            str(VAULT_DIR / "**" / "*.md"),
            str(BASE_DIR / "*.pdf"),
            str(BASE_DIR / "*.md"),
            str(BASE_DIR / "output" / "*"),
            str(VAULT_DIR / "**" / "*.docx"),
            str(BASE_DIR / "*.docx"),
        ]:
            for fp in _glob.glob(pattern, recursive=True):
                p = Path(fp)
                if p.name == "INDEX.md":
                    continue
                age = (datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)).total_seconds()
                if age < 120:  # created in the last 2 minutes
                    recent_files.append(p)

        # Send text reply
        if reply:
            if len(reply) > 4000:
                for i in range(0, len(reply), 4000):
                    await update.message.reply_text(reply[i:i+4000])
            else:
                await update.message.reply_text(reply)

        # Send any recently created files
        for fpath in recent_files:
            try:
                with open(fpath, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=fpath.name,
                        caption=f"📄 {fpath.name}"
                    )
            except Exception as e:
                log.warning(f"TELEGRAM: failed to send file {fpath}: {e}")

    async def _handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not _tg_allowed(uid):
            await update.message.reply_text("Not authorized.")
            return
        _tg_sessions[uid] = []
        vault_count = len(list(VAULT_DIR.rglob("*.md"))) if VAULT_DIR.exists() else 0
        habits_count = len(load_habits().get("routes", {}))
        await update.message.reply_text(
            f"◆ {vessel_name}\n"
            f"  vault: {vault_count} notes | habits: {habits_count} routes\n\n"
            f"Send me anything. /clear to reset. /status for diagnostics."
        )

    async def _handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not _tg_allowed(uid):
            return
        _tg_sessions[uid] = []
        sid = f"tg_{uid}"
        if sid in _chat_sessions:
            del _chat_sessions[sid]
        await update.message.reply_text("Session cleared.")

    async def _handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not _tg_allowed(uid):
            return
        vctx = load_vessel()
        vault_count = len(list(VAULT_DIR.rglob("*.md"))) if VAULT_DIR.exists() else 0
        habits = load_habits()
        routes_count = len(habits.get("routes", {}))
        await update.message.reply_text(
            f"◆ {vessel_name}\n"
            f"  provider: {LLM_PROVIDER} | model: {BRIDGE_MODEL}\n"
            f"  vault: {vault_count} notes\n"
            f"  habits: {routes_count} routes\n"
            f"  state: {len(vctx.get('state', '') or '')} chars"
        )

    async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not _tg_allowed(uid):
            await update.message.reply_text("Not authorized.")
            return

        user_text = update.message.text
        if not user_text:
            return

        session_id = f"tg_{uid}"
        history = _tg_sessions.get(uid, [])
        history.append({"role": "user", "content": user_text})
        _tg_sessions[uid] = history
        _chat_sessions[session_id] = history

        try:
            vctx = load_vessel()
            route = hecate(vctx, user_text)
            tree_context = build_tree_context(route)
            system = _build_chat_system(
                vctx["vessel"],
                vctx["state"] or "(no prior state)",
                tree_context
            )

            # Override the system prompt ending for telegram context
            system = system.replace(
                "You are in a direct terminal conversation with your operator.",
                "You are in a Telegram conversation with your operator. "
                "You have FULL tool access — read files, write files, run commands. "
                "All tools are auto-approved. "
                "IMPORTANT: When asked to create a PDF, paper, report, or document, "
                "ALWAYS use the create_pdf tool. Read the source content first with vault_read, "
                "then call create_pdf with the full content. The PDF will be auto-sent to Telegram. "
                "Write the FULL content — never truncate, never summarize, never abbreviate. "
                "Long messages will be split automatically. Use plain text, no ANSI codes. Be direct and thorough."
            )

            result = await _operator_loop(session_id, history, system)

            if result["done"]:
                reply = result["reply"]

                # Record habit success
                try:
                    habits = load_habits()
                    nodes = route.get("nodes", [])
                    record_success(habits, _make_task_key(user_text),
                                   _extract_signature(user_text), nodes)
                except Exception:
                    pass

                # Trim if needed
                if len(history) >= CHAT_HISTORY_MAX:
                    history = await _summarize_and_compress(session_id, history, vctx["vessel"])
                    _tg_sessions[uid] = history
                    _chat_sessions[session_id] = history
                _save_chat_history(session_id, history)

                await _tg_send_reply(update, reply, vessel_name)

            else:
                # Dangerous tools — auto-approve in Telegram (operator is messaging directly)
                vessel_text = result.get("vessel_text", "")
                pending = result.get("pending", [])
                action_names = [a.get("type", "?") for a in pending]
                if vessel_text:
                    await update.message.reply_text(vessel_text)

                # Execute all pending actions
                pending_data = _chat_pending.pop(session_id, None)
                if pending_data:
                    tool_results = pending_data["safe_results"]
                    for tc in pending_data["dangerous_calls"]:
                        res = _exec_dangerous_tool(tc["name"], tc["input"])
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": res
                        })
                    history.append({"role": "user", "content": tool_results})
                    result2 = await _operator_loop(session_id, history, pending_data["system"])
                    if result2["done"]:
                        reply = result2["reply"]
                        await _tg_send_reply(update, reply, vessel_name)

        except Exception as e:
            log.error(f"TELEGRAM error: {e}")
            await update.message.reply_text(f"Error: {type(e).__name__}: {str(e)[:200]}")
            if history and history[-1].get("role") == "user":
                history.pop()

        # Vault commit — every 10 messages, check novelty and save
        if len(history) >= 4 and len(history) % 4 == 0:
            try:
                convo_lines = []
                for msg in history[-20:]:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                    convo_lines.append(f"{msg.get('role', '?')}: {str(content)[:300]}")
                convo_text = "\n".join(convo_lines)
                hrr = HolographicMemory(path=str(VAULT_DIR / "hrr_memory.json"))
                novelty = hrr.novelty(convo_text)
                if novelty >= 0.4:
                    resp = await asyncio.to_thread(lambda: client.messages.create(
                        model=BRIDGE_MODEL_FAST, max_tokens=400,
                        messages=[{"role": "user", "content":
                            f"Summarize this Telegram conversation into a vault note. "
                            f"Key decisions, topics, action items. Use [[wikilinks]]. Under 200 words.\n\n{convo_text}"}],
                        timeout=60,
                    ))
                    note_content = _get_text(resp)
                    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
                    note_path = VAULT_DIR / "sessions" / f"tg_{vessel_name}_{ts}.md"
                    note_path.parent.mkdir(parents=True, exist_ok=True)
                    note_path.write_text(
                        f"---\ntags: [session, telegram, {vessel_name.lower()}]\n"
                        f"vessel: {vessel_name}\nuser: {uid}\n"
                        f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
                        f"novelty: {novelty:.2f}\n---\n\n"
                        f"# Telegram Session\n\n{note_content}"
                    )
                    hrr.bind(f"tg_{session_id}_{len(history)}", convo_text[:500],
                             metadata={"vessel": vessel_name, "channel": "telegram"})
                    log.info(f"TELEGRAM: vault commit for user {uid} (novelty: {novelty:.2f})")
            except Exception as e:
                log.warning(f"TELEGRAM vault commit failed: {e}")

    # Build the bot
    app_tg = Application.builder().token(TELEGRAM_TOKEN).build()
    app_tg.add_handler(CommandHandler("start", _handle_start))
    app_tg.add_handler(CommandHandler("clear", _handle_clear))
    app_tg.add_handler(CommandHandler("status", _handle_status))
    app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

    return app_tg, vessel_name


async def _start_telegram_standalone():
    """Run telegram as standalone (no HTTP)."""
    app_tg, vessel_name = await _telegram_run()

    loop = asyncio.get_event_loop()
    loop.create_task(_heartbeat_loop())
    loop.create_task(_substrate_loop())

    log.info(f"TELEGRAM: {vessel_name} is live. Polling...")
    await app_tg.initialize()
    await app_tg.start()
    await app_tg.updater.start_polling()

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        await app_tg.updater.stop()
        await app_tg.stop()
        await app_tg.shutdown()


if __name__ == "__main__":
    mode = "cli"
    if "--serve" in sys.argv:
        mode = "serve"
    elif "--telegram" in sys.argv:
        mode = "telegram"

    if mode == "serve":
        # HTTP + Telegram (auto-starts Telegram if token exists in .env)
        if not (VESSEL_DIR / "VESSEL.md").exists() and sys.stdin.isatty():
            _init_dirs()
            _auto_init_tty()
            _generate_default_tree()
            _init_vault()
        if TELEGRAM_TOKEN:
            log.info("TELEGRAM token found — bot will start alongside HTTP server")
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
    elif mode == "telegram":
        # Telegram only (no HTTP server)
        if not TELEGRAM_TOKEN:
            print("\033[91m  error: TELEGRAM_BOT_TOKEN not set in .env\033[0m")
            print("\033[90m  1. Message @BotFather on Telegram\033[0m")
            print("\033[90m  2. /newbot → pick a name → get token\033[0m")
            print("\033[90m  3. Add to .env: TELEGRAM_BOT_TOKEN=your_token\033[0m")
            print("\033[90m  4. Optional: TELEGRAM_ALLOWED_IDS=12345,67890\033[0m")
            sys.exit(1)
        asyncio.run(_start_telegram_standalone())
    else:
        # Terminal REPL mode (default)
        asyncio.run(_cli_loop())
