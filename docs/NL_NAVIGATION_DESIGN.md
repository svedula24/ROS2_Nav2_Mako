# Natural-language navigation: technical design

Status: **draft for review**. Requirements: [NL_NAVIGATION.md](NL_NAVIGATION.md).
Target: the splat demo (branch `3dgs-sim-nl-navigation`), with every model running on the dev laptop.
Phase 1 (tier 0, no models) is implemented; tiers 1–2 below are still the design.

## 1. Overview

NL navigation is a **ROS package, `src/m20_nlnav`**, that sits in front of the mission server,
the same way the `tools/dev/m20` CLI does. It follows the repo's existing workflow:
- it's a node in the stack;
- dev bringup hot-loads its Python from the repo;
- production runs it as a compose service.

```
Viser panel ─┐                         ┌─ /detections/objects (object map, JSON)
m20 say ─────┼─ NavigateNL action ──▶ nlnav_node ◀─ /odom_true (pose), /global_costmap/costmap
(prod client)┘  /m20/nlnav/navigate    │   ◀─ scene yaml waypoints:
                (feedback, cancel)     │ tier 0: exact match        (Phase 1)
                                       │ tier 1: local LLM → query  (Phase 2)
                                       │ resolve: geometry + rank, tier 2 VLM (Phase 3)
                                       │ approach pose on the costmap
                                       ▼
                         RunMission (N × TYPE_GOTO) → m20_missions → Nav2
                         /m20/nlnav/markers (Foxglove / RViz)
```

Design rules:
1. **Language never drives the robot directly.**
   - The parser (later, the LLM) outputs a structured **query**.
   - Plain code resolves it against what the robot has actually seen and produces goal poses.
   - Those poses go through the existing mission server, Nav2 and commander, unchanged
     (ADR-014, ADR-005).
2. **One entry point for every caller.** Viser, the CLI and production clients all call the
   same action. The node executes through `RunMission` and forwards feedback and cancel.
   - `dry_run` returns the plan without driving, for dev preview and confirm.
   - A new command cancels the node's previous mission. `stop` cancels it outright.
   - Callers that start missions themselves (the Viser explore tour) must stop them first;
     the viewer does this.
3. **Testable without the stack.** The logic modules (`world`, `tier0`, `resolve`,
   `approach`) import no rclpy.
   - **Fast CI job:** plain pytest on the logic.
   - **ROS CI job:** `colcon test` runs `src/m20_nlnav/test/test_node.py` with real rclpy, a
     fake mission server and fixture topics.

## 2. Components and files

| File | Role |
|---|---|
| `src/m20_msgs/action/NavigateNL.action`, `msg/NLCandidate.msg` | The interface (§3) |
| `src/m20_nlnav/m20_nlnav/world.py` | World model: objects, waypoints, pose, cost grid; detector classes + synonyms |
| `src/m20_nlnav/m20_nlnav/tier0.py` | Exact-match parser (Phase 1). Returns None for anything it can't fully parse |
| `src/m20_nlnav/m20_nlnav/resolve.py` | Candidates, ranking, dev/demo ambiguity policy, multi-step chaining |
| `src/m20_nlnav/m20_nlnav/approach.py` | Approach-pose search on the costmap |
| `src/m20_nlnav/m20_nlnav/nlnav_node.py` | ROS node: subscriptions, NavigateNL server, RunMission client, markers, health |
| `src/m20_nlnav/test/test_node.py` | ROS-level test (colcon test) |
| `tests/test_nlnav_{tier0,resolve,approach}.py` | Fast logic tests |
| `tools/nav/bringup_plan_a.sh` | Hot-loads the node into the commander container (`M20_NLNAV`, default 1) |
| `docker/compose.yaml` | `nlnav` service (prod) |
| `tools/dev/m20` | `m20 say "..." [--dev] [--dry-run] [--pick ID]` |
| `tools/splat/splat_viewer.py` | "Navigate (natural language)" panel, plan highlight in 3D + mini-map |
| `tools/sim/indoor_splat*.scene.yaml` | `waypoints:` block (start, stop 1–13) |
| `tools/splat/extract_geometry.py` | Preserves `waypoints:` across re-extraction (while the frame is unchanged) |
| Later: `m20_nlnav/{backends,parse}.py`, `prompts/`, `eval/` | Tiers 1–2 (LLM / VLM via Ollama), eval set |
| Later: `tools/splat/detector_node.py` | Per-object crops + keyframes (tier 2, open-vocabulary search) |

**Rebuilds:**
- Python changes need **no image rebuild**: bringup copies `src/m20_nlnav/m20_nlnav` into
  `/cfg` and runs it, like the behaviour engine.
- Interface changes (`m20_msgs`) do. Rebuild `m20_autonomy` → `m20_sim` → `m20_splat` (the
  viewer imports the action) with `bash tools/splat/setup_splat_demo.sh --rebuild`.

## 3. Interface: `m20_msgs/action/NavigateNL` (`/m20/nlnav/navigate`)

```
# goal
string text
uint8 MODE_DEV=0          # ambiguity -> status "ask" with candidates (human picks)
uint8 MODE_DEMO=1         # ambiguity -> highest-confidence match
uint8 mode
bool dry_run              # resolve only (preview / confirm); false = resolve + drive
int32 pick_id             # dev follow-up: the chosen candidate's object id; -1 = none
---
bool success
string status             # plan | ask | not_found | error | done | canceled | stopped
string interpretation     # "printer #14 (0.82) → stop 3"
m20_msgs/MissionItem[] items         # the resolved goals (map frame)
m20_msgs/NLCandidate[] candidates    # object_id, cls, position, score, why
uint8 tier
float32 latency_ms        # time to resolve (not to drive)
string detail             # reason / help text
---
string stage              # resolving | navigating
uint32 current_index      # forwarded from RunMission feedback
string current_label
string status
```

**Dev flow:**
1. Send `dry_run` → get `plan` or `ask`, and highlight it.
2. Re-send with `pick_id` if the answer was `ask`.
3. Send with `dry_run=false` to drive.

**Demo flow:** send with `dry_run=false` straight away.

**Node parameters:** `scene_yaml`, `min_score` (0.5), `odom_topic` (`/odom_true`), `objects_topic`,
`costmap_topic` (fallback), `costmap_service` (`/global_costmap/get_costmap`, queried per command).

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
