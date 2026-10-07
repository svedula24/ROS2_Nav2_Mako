# Natural-language navigation: technical design

Status: **draft for review**. Requirements: [NL_NAVIGATION.md](NL_NAVIGATION.md).
Target: the splat demo on branch `3dgs-sim-env`, with every model running on the dev laptop.

## 1. Overview

```
Viser text box ──/nlnav/request──▶ nlnav_node ──/nlnav/response──▶ Viser
                                      │                             │ (dev: user picks / confirms;
          world model ◀───────────────┤                             │  demo: auto)
   /detections/objects (object map)   │ tier 0: exact match          ▼
   scene yaml waypoints:              │ tier 1: local LLM → query  RunMission (N × TYPE_GOTO)
   /odom_true (robot pose)            │ resolve: geometry + rank      │
   /global_costmap/costmap            │ tier 2: VLM attribute check   ▼
   object crops + keyframes (disk)    │ approach pose (costmap)    mission server → Nav2
                                      ▼
                         Ollama (local LLM + VLM, GPU)
```

Design rules:
1. **Language never drives the robot directly.**
   - The LLM outputs a structured **query**.
   - Plain code resolves the query against what the robot has actually seen and produces goal
     poses.
   - Those poses go through the existing mission server, Nav2 and commander, unchanged.
2. **Understanding and executing are separate.**
   - `nlnav_node` takes text in and returns a **plan** (goal poses plus explanation).
   - The Viser viewer executes the plan using its existing `RunMission` client (`send()` in
     `tools/splat/splat_viewer.py`), so it stays the **only goal sender** in the demo and can't
     race the explore tour.
   - In production, a thin executor replaces the viewer.
3. **A ROS-free core.** Parsing, resolving and approach-pose code is plain Python (numpy only),
   unit-tested in CI with a fake LLM. `nlnav_node.py` is a thin ROS wrapper.

## 2. Components and files

| File | Role |
|---|---|
| `tools/nlnav/schema.py` | Query / plan dataclasses + JSON schema (built per request with live enums) |
| `tools/nlnav/world.py` | `WorldSnapshot`: objects, waypoints, robot pose, costmap; synonyms → detector classes |
| `tools/nlnav/tier0.py` | Exact / fuzzy matcher for single-target commands and stop/cancel |
| `tools/nlnav/backends.py` | `Backend` interface; `OllamaBackend` (default), `AnthropicBackend` (optional, off), `FakeBackend` (tests) |
| `tools/nlnav/parse.py` | Tier routing: tier 0 → tier 1 LLM → validate → (fallback) bigger model |
| `tools/nlnav/resolve.py` | Candidate filtering (class, relations, ordinal), tier 2 attribute check, ranking, ambiguity policy |
| `tools/nlnav/approach.py` | Approach-pose search on the costmap |
| `tools/nlnav/prompts/` | System prompts + few-shot examples (versioned text files) |
| `tools/nlnav/nlnav_node.py` | ROS wrapper: subscriptions, request/response topics, timing |
| `tools/nlnav/eval/` | Golden command set + saved object-map snapshot + `run_eval.py` |
| `tools/splat/splat_viewer.py` | "Navigate" GUI folder, multi-item `send()`, target highlight, plan execution |
| `tools/splat/detector_node.py` | Per-object best crops + keyframe store (for tier 2 and open-vocabulary search) |
| `tools/sim/indoor_splat.scene.yaml` | New `waypoints:` block (names → x, y, yaw) |
| `tools/splat/extract_geometry.py` | Preserve `waypoints:` across re-extraction (same as the `topdown:` block, l.339) |
| `tools/splat/splat_sim_up.sh` | `M20_NLNAV=1` (default on): start the Ollama + nlnav containers |
| `tests/test_nlnav_*.py` | Unit tests (ROS-free, run in CI) |

All new runtime code runs in the existing `m20_splat` image: it has rclpy, numpy, torch and
OWLv2, and the Ollama client uses only `urllib`. **No image rebuild** is needed. Ollama runs in
the stock `ollama/ollama` container.

## 3. Interfaces

The interfaces use JSON on `std_msgs/String`, the same pattern as `/detections/objects`. This
avoids an `m20_msgs` change and an image rebuild while the design settles. **Prod:** promote
them to an `m20_msgs/action/NavigateNL`, with goal = text, feedback = plan/step and
result = outcome.

**`/nlnav/request`**
```json
{"id": "r42", "text": "go to the chair next to the whiteboard", "mode": "dev",
 "pick": null}
```
`pick` holds the user's choice from the candidates in a dev-mode `ask` response. The text is
re-resolved with that choice fixed.

**`/nlnav/response`**
```json
{"id": "r42", "status": "plan",            // plan | ask | not_found | error | stopped
 "tier": 1, "interpretation": "office chair #7 (near whiteboard #3, 0.9 m)",
 "steps": [{"label": "office chair #7",
            "target": {"kind": "object", "id": 7, "cls": "office chair", "xyz": [3.1, -4.2, 0.5]},
            "goal": [2.3, -4.0, -0.24], "confidence": 0.81,
            "candidates": [{"id": 7, "score": 0.81, "why": "0.9 m from whiteboard"},
                           {"id": 9, "score": 0.74, "why": "1.4 m from whiteboard"}]}],
 "latency_ms": {"tier0": 2, "llm": 640, "vlm": 0, "resolve": 8, "total": 655},
 "message": "Going to the office chair next to the whiteboard."}
```

## 4. Structured query (what the LLM must output)

The schema is generated **per request**, with live enums. `class` may only be one of the classes
currently in the object map (plus `"unknown"`), and `name` may only be a waypoint that exists.
Ollama's `format` (JSON schema) constrains decoding to that schema, so the LLM **cannot invent**
a class or waypoint.

```json
{"steps": [
  {"action": "goto",                          // goto | stop | return
   "target": {
     "type": "object",                        // object | waypoint | start | here_at_command
     "class": "office chair",                 // enum: live classes ∪ {"unknown"}
     "noun": "chair",                         // the user's word (used for "unknown" → open-vocab)
     "attributes": ["red"],                   // free text, checked by the VLM
     "relations": [{"rel": "near", "ref": {"class": "whiteboard", "attributes": []}}],
                                              // rel: near | left_of_robot | right_of_robot | between
     "select": {"order": "nearest", "index": 0},   // nearest | farthest | left_to_right | right_to_left
     "name": null}}                           // waypoint name when type = waypoint
]}
```

Validation, in code:
- Every step is checked against the snapshot.
- The `ref` classes must exist.
- Indexes must be in range.
- If validation fails, the request is retried once with the error fed back, then escalated to the
  fallback model, then answered `not_found` with the reason.

## 5. Resolution

1. **World snapshot.** Taken when the request arrives:
   - confirmed objects (`id, cls, xyz, size, score, n`; `detector_node.py:ObjectMap`), filtered
     at the viewer's Min confidence;
   - waypoints from the scene yaml;
   - the robot pose from `/odom_true`;
   - the latest global costmap.
2. **Class filter.** Keep candidates of `class`. Synonyms are handled in `world.py` before
   tier 0 ("chair" → "office chair", "bin" → "trash can", "screen" → "computer monitor").
3. **Relations** (geometry on object centres and footprints):
   - **`near`:** footprint distance to any ref object ≤ 1.5 m; ranked by that distance.
   - **`left_of_robot` / `right_of_robot`:** the sign of the cross product against the robot's
     heading **at command time**. Whether it should instead mean a fixed map direction is an open
     question in the requirements.
   - **`between`:** the candidate projects inside segment A–B, within 1.5 m of it.
4. **Select.**
   - "nearest"/"farthest" sort by path-free Euclidean distance from the robot. Using Nav2
     `ComputePathToPose` is a later refinement.
   - "second … on the left" filters to the left side, then sorts by forward distance.
5. **Attributes (tier 2).**
   - For each remaining candidate (max 6), the VLM is asked for a yes/no plus confidence on the
     object's best crop:
     `{"match": bool, "confidence": 0-1, "evidence": "..."}`.
   - Candidates scoring `match=false` or `confidence < 0.5` are dropped.
   - If an object has no crop, it's flagged "unverified":
     - dev mode: `ask`;
     - demo mode: rank it below the verified candidates.
6. **Unknown class (open-vocabulary).**
   - If `class == "unknown"`, run OWLv2 with the text query `noun` over the stored keyframes.
   - Hits are lifted to 3D with the saved depth and pose (`box3d()`, `detector_node.py:85`) and
     merged like the object map.
   - If anything is found, it becomes a temporary candidate; otherwise `not_found` ("I haven't
     seen a coffee machine").
7. **Rank and decide.**
   - Score = detector score × relation fit × VLM confidence (missing factors count as 1).
   - **Demo mode:** take the top candidate.
   - **Dev mode:** if the top two scores are within 0.1 of each other or there are > 1
     candidates with no relation or ordinal to narrow them, return `ask` with up to 5 candidates.
8. **Approach pose** (`approach.py`):
   - Sample 24 headings × radii {0.9, 1.2, 1.5 m} around the object footprint.
   - Keep poses whose robot footprint (0.86 × 0.56 m, padded) has max cost < 50 on the global
     costmap. That threshold sits inside the keep-away inflation, so it matches how the lap
     drives.
   - Prefer the side facing the robot's approach direction, then the shortest distance.
   - Yaw faces the object centre.
   - If nothing is valid, return `not_found` with "can't get close to printer #14 safely".
9. **Waypoints and specials.**
   - `type: waypoint` → the yaml pose.
   - `start` → spawn (0, 0, 0).
   - `here_at_command` → the robot pose at request time. This is the default meaning of
     "come back" until the open question is answered.

## 6. Models and backends

```python
class Backend(Protocol):
    def complete_json(self, system: str, user: str, schema: dict, timeout_s: float) -> dict: ...
    def check_image(self, image_jpeg: bytes, question: str, schema: dict, timeout_s: float) -> dict: ...
```

| Role | Proposed model (via Ollama, quantised) | Approx. VRAM |
|---|---|---|
| Tier 1 parser | Qwen2.5-7B-Instruct (Q4_K_M) | ~5 GB |
| Tier 2 VLM + parser fallback | Qwen2.5-VL-7B (Q4) | ~6 GB |
| Optional cloud | Claude (Anthropic API), off by default | — |

- The model names and quantisations are **to be confirmed** when this is built. Pick by the eval
  set's accuracy and latency, not by name.
- **GPU budget:** the laptop GPU has 24 GB, and about 3 GB is in use with the demo stopped. The
  splat camera, the viewer renders and OWLv2 all share it.
  - Measure the demo's footprint first.
  - Ollama `keep_alive` keeps the parser resident and loads the VLM on demand. If memory is
    tight, use the VL model for both roles (one ~6 GB model).
- **Jetson (prod):** tiers 0–1 are fine with a 3–7B model on Orin. Tier 2 is the risk: run it
  less often (only for attribute commands), use a smaller VLM, or offload it.

## 7. Detector additions (crops and keyframes)

- **Best crops:**
  - For each tracked object, keep the top 3 crops by `score × box area` (box padded 15 %, JPEG,
    ≤ 384 px).
  - Save them to `~/m20_sim/nlnav/crops/<id>_<k>.jpg`.
  - Add `"crops": [...]` to the object JSON.
- **Keyframes:**
  - Every ≥ 1 m of travel or ≥ 30° of rotation, save the RGB (JPEG), depth (float16 `.npy`) and
    pose to `~/m20_sim/nlnav/keyframes/`. A lap is about 150 keyframes, roughly 60 MB.
  - Cleared at launch, like the object map.
- The new host directory is mounted into the detector and nlnav containers.

## 8. Viser UI ("Navigate" folder)

- **Controls:** a text box, **Go**, **Stop**, and a mode dropdown (Dev / Demo; the default is
  Demo when `M20_EXPLORE=1`).
- **Interpretation line:** `interpretation` plus the tier and latency.
- **Highlighting:**
  - The chosen target's 3D box is drawn thicker and in magenta, with a label.
  - The goal pose is shown as an arrow in 3D and on the mini-map.
  - Dev `ask`: each candidate gets a numbered label in 3D and on the mini-map, plus one button
    per candidate.
- **Execution:**
  - On `plan`: in demo mode it executes immediately; in dev mode a **Confirm** button appears.
  - `send()` is generalised from one item to N items (one `RunMission`).
  - Starting a plan stops the explore tour first.
- **History:** the last 5 commands with their outcome.

## 9. Testing

- **Unit tests** (CI, no ROS, no models):
  - `tests/test_nlnav_tier0.py`: names, synonyms, stop.
  - `tests/test_nlnav_resolve.py`: near/left/between/ordinal on a fixture map.
  - `tests/test_nlnav_approach.py`: free/blocked costmap fixtures.
  - `tests/test_nlnav_parse.py`: `FakeBackend` returns canned JSON; checks validation, retry and
    escalation.
- **Eval** (local, real models), `tools/nlnav/eval/run_eval.py`:
  - ≥ 20 commands in `commands.yaml`, each with the expected target id(s), run against a saved
    object-map snapshot, crops and keyframes from one lap.
  - Reports parse accuracy, correct-target rate, wrong-target count (must be 0) and p50/p95
    latency per tier.
- **Live:** in the demo, run the 5-command script from the requirements §6, then a normal lap
  (13/13, 0 falls).

## 10. Build phases

| Phase | Delivers | Demo-able |
|---|---|---|
| 1 | `waypoints:`, world snapshot, tier 0, approach pose, nlnav node, Viser Navigate panel, multi-item execute | "go to the printer", "go to stop 3", "stop" |
| 2 | Tier 1 LLM (Ollama), schema with live enums, relations, ordinals, multi-step, dev/demo ambiguity | "the chair next to the whiteboard", "printer then ladder then back" |
| 3 | Detector crops, tier 2 VLM attribute checks | "the red chair", "the open door" |
| 4 | Keyframes + open-vocabulary search | "the coffee machine" |
| 5 (later) | `NavigateNL` action, edge-device client, Jetson backends | Prod |

Each phase ends with its unit tests green, the eval numbers recorded here, and one clean demo
lap.

## 11. Risks

- **Object-map quality limits everything.**
  - Duplicate or misplaced objects become wrong targets.
  - Mitigation: the confirmation threshold (n ≥ 3), the Min confidence filter, and dev-mode
    review.
- **Splat image quality vs the VLM.** Crops of splat-blurred objects may defeat attribute checks
  (e.g. colour of a smeared chair). Measure this in phase 3 before promising attribute commands
  in the demo.
- **GPU contention:** LLM and VLM bursts can slow the photoreal renders. Measure; cap model
  sizes; load the VLM on demand.
- **Left/right ambiguity:** a wording decision, not a technical one. Pin it in the requirements.
