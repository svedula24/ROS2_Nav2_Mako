"""Instruction -> structured query, via an LLM with structured output (offline phase, NL_NAV architecture).

The LLM's ONLY job is to fill the schema below; finding the object and choosing the waypoint are plain code
(semantic.py) reading the object map, so the LLM never picks the goal and the same query always gives the
same waypoint. Output that doesn't validate is refused ("please rephrase"); nothing moves.

  {"steps": [{"target": "valve", "anchor": "pump", "relation": "near", "pick": "best"}], "stop": false}

Backends (one interface, NLNAV_LLM selects): ollama (default, local), claude (optional), json (the instruction
IS a JSON query: tests, scripted use). No backend may change the schema.
"""
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

RELATIONS = ["near", "left_of", "right_of", "in_front_of", "behind"]
PICKS = ["best", "nearest"]

SCHEMA = {
    "type": "object",
    "properties": {                       # order matters: the model commits to `understood` before any steps
        "understood": {"type": "boolean"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "anchor": {"type": ["string", "null"]},
                "relation": {"type": ["string", "null"], "enum": RELATIONS + [None]},
                "pick": {"type": "string", "enum": PICKS},
            },
            # all required (null when unused): Ollama's schema grammar skips optional keys, dropping anchors
            "required": ["target", "anchor", "relation", "pick"]}},
        "stop": {"type": "boolean"},
    },
    "required": ["understood", "steps", "stop"],
}

SYSTEM = """You convert a robot navigation instruction into JSON for an indoor robot. Output ONLY the JSON.
Fields:
- understood: false if the text is NOT a navigation or stop instruction (gibberish, random letters, chit-chat,
  a question, an unrelated request); then steps is []. Never invent a destination the text doesn't name.
- steps: one entry per place to go, in order ("then", "and then", "after that" start a new step).
  - target: the thing or place to go to, as a short noun phrase WITHOUT articles, keeping only the words
    that identify it (e.g. "fire extinguisher", "yellow tank", "ladder"). Rewrite descriptions into the
    object's common name ("the thing you use to put out fires" -> "fire extinguisher").
  - anchor: a second object the target is described relative to, else null.
  - relation: how the target relates to the anchor: near | left_of | right_of | in_front_of | behind, else null.
  - pick: "nearest" if the user says nearest/closest, else "best".
- stop: true only if the user asks to stop/halt/cancel (then steps is []).
Special targets: going back to where the robot started / home / the dock -> the named place given below
if one matches, else "start". "come back" / "return" as a LATER step -> "back".
"next to", "beside", "by", "close to" all mean near; left_of / right_of are ONLY for "to the left/right of".
Named places on this site: {places}.
Examples:
"go to the ladder then come back to the start" ->
 {"understood": true, "steps": [{"target": "ladder", "anchor": null, "relation": null, "pick": "best"},
  {"target": "start", "anchor": null, "relation": null, "pick": "best"}], "stop": false}
"find the valve right next to the pump and then return" ->
 {"understood": true, "steps": [{"target": "valve", "anchor": "pump", "relation": "near", "pick": "best"},
  {"target": "back", "anchor": null, "relation": null, "pick": "best"}], "stop": false}
"the chair to the left of the printer" ->
 {"understood": true, "steps": [{"target": "chair", "anchor": "printer", "relation": "left_of", "pick": "best"}],
  "stop": false}
"the nearest trash can" (a bare noun phrase IS an instruction: go there) ->
 {"understood": true, "steps": [{"target": "trash can", "anchor": null, "relation": null, "pick": "nearest"}],
  "stop": false}
"blorp" -> {"understood": false, "steps": [], "stop": false}"""


@dataclass
class QueryStep:
    target: str
    anchor: Optional[str] = None
    relation: Optional[str] = None
    pick: str = "best"


@dataclass
class Query:
    steps: List[QueryStep] = field(default_factory=list)
    stop: bool = False


class NotUnderstood(Exception):
    pass


class ParserUnavailable(NotUnderstood):
    """The backend failed (network, timeout, model not loaded): our fault, not the user's wording."""


# "stop" must never wait on (or fail with) the LLM: a plain-code match runs first (REQ: stop needs nothing).
_STOP_RE = re.compile(r"^\s*(please\s+)?(stop|halt|cancel|abort)(\s+(it|now|moving|that|the robot|the mission))?"
                      r"[\s.!]*$", re.IGNORECASE)


def is_stop(text: str) -> bool:
    return bool(_STOP_RE.match(text or ""))


def validate(obj) -> Query:
    """Strict schema check -> Query, else NotUnderstood (with the reason)."""
    if not isinstance(obj, dict) or not isinstance(obj.get("stop"), bool) or not isinstance(obj.get("steps"), list):
        raise NotUnderstood("the parser returned no valid query")
    if obj.get("understood", True) is False and not obj["stop"]:     # optional: hand-written JSON queries omit it
        raise NotUnderstood("that doesn't look like a navigation instruction")
    if obj["stop"]:
        return Query([], True)
    if not obj["steps"]:
        raise NotUnderstood("no destination found in the instruction")
    steps = []
    for s in obj["steps"]:
        if not isinstance(s, dict):
            raise NotUnderstood("malformed step")
        target = str(s.get("target") or "").strip().lower()
        target = re.sub(r"^(the|a|an|my)\s+", "", target)
        if not target:
            raise NotUnderstood("a step has no target")
        rel = s.get("relation")
        anchor = s.get("anchor")
        if rel not in RELATIONS + [None]:
            raise NotUnderstood(f"unknown relation {rel!r}")
        anchor = re.sub(r"^(the|a|an|my)\s+", "", str(anchor).strip().lower()) if anchor else None
        if rel and not anchor:
            rel = None            # a relation with nothing to relate to carries no information: drop it, keep the
                                  # step (qwen2.5:3b answers "the nearest trash can" with relation "near", anchor null)
        pick = s.get("pick", "best")
        if pick not in PICKS:
            raise NotUnderstood(f"unknown pick {pick!r}")
        steps.append(QueryStep(target, anchor, rel if anchor else None, pick))
    return Query(steps, False)


# ------------------------------------------------------------------ backends
class JsonBackend:
    """The instruction is already a JSON query (tests, scripted use)."""
    name = "json"

    def complete_json(self, system, user, schema, timeout_s=5.0):
        try:
            return json.loads(user)
        except ValueError as e:
            raise NotUnderstood(f"not JSON: {e}")


class OllamaBackend:
    """Local model via Ollama's /api/chat with a JSON-schema `format` (constrained decoding)."""
    name = "ollama"

    def __init__(self, url=None, model=None):
        self.url = (url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.environ.get("NLNAV_LLM_MODEL", "qwen2.5:7b-instruct")

    KEEP_ALIVE = "12h"                        # a cold load is 15-20 s; keep the model resident for the session

    def complete_json(self, system, user, schema, timeout_s=20.0):
        body = {"model": self.model, "stream": False, "format": schema, "options": {"temperature": 0},
                "keep_alive": self.KEEP_ALIVE,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            content = json.loads(r.read())["message"]["content"]
        return json.loads(content)

    def warm(self, timeout_s=120.0):
        """Load the model into memory (an empty chat loads it without generating)."""
        body = {"model": self.model, "messages": [], "keep_alive": self.KEEP_ALIVE}
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=timeout_s).read()


class ClaudeBackend:
    """Anthropic Messages API, forced tool call whose input_schema is the query schema (opt-in)."""
    name = "claude"

    def __init__(self, model=None):
        self.key = os.environ["ANTHROPIC_API_KEY"]
        self.model = model or os.environ.get("NLNAV_LLM_MODEL", "claude-haiku-4-5-20251001")

    def complete_json(self, system, user, schema, timeout_s=20.0):
        body = {"model": self.model, "max_tokens": 512, "system": system,
                "tools": [{"name": "navigation_query", "description": "The structured navigation query.",
                           "input_schema": schema}],
                "tool_choice": {"type": "tool", "name": "navigation_query"},
                "messages": [{"role": "user", "content": user}]}
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json", "x-api-key": self.key,
                                              "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            msg = json.loads(r.read())
        return next(b["input"] for b in msg["content"] if b.get("type") == "tool_use")


def make_backend(name=None):
    name = (name or os.environ.get("NLNAV_LLM", "ollama")).lower()
    return {"ollama": OllamaBackend, "claude": ClaudeBackend, "json": JsonBackend}[name]()


def parse(text: str, backend, places=("start",), timeout_s=20.0) -> Query:
    """Instruction -> validated Query. Raises NotUnderstood for anything that isn't a clean query."""
    system = SYSTEM.replace("{places}", ", ".join(places) or "none")
    try:
        raw = backend.complete_json(system, text.strip(), SCHEMA, timeout_s)
    except NotUnderstood:
        raise
    except Exception as e:                    # network, timeout, bad JSON from the model
        raise ParserUnavailable(f"parser unavailable or returned bad output ({type(e).__name__}: {e})")
    return validate(raw)
