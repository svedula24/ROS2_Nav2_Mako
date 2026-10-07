# Natural-language navigation: requirements

Status: **draft**. Built from the 2026-10-07 Q&A; edit freely.
Technical design: [NL_NAVIGATION_DESIGN.md](NL_NAVIGATION_DESIGN.md)

## 1. Goal

Tell the robot where to go in plain English, and it drives there:

> "go to the printer", "go to stop 3", "the chair next to the whiteboard",
> "the red chair", "go to the printer, then the ladder, then come back"

The feature is built and demoed first in the **splat demo**
([SPLAT_DEMO.md](SPLAT_DEMO.md)), with a text box in **Viser**. The core is kept separate from
Viser so a production interface can be added later.

### In scope (now)
- A command box in Viser, plus feedback on what the robot understood and where it's going.
- Commands that target detected objects, named waypoints, relations, attributes and multi-step routes.
- Every model runs **locally on the dev laptop** (RTX 5090 Laptop, 24 GB). Network access is
  allowed (e.g. to download models), but nothing calls the cloud by default.

### Out of scope (later)
- Production interfaces: an edge device / operator app, and a public ROS action for external callers.
- On-robot (Jetson) deployment. The design must not block it (see §5).
- Voice input.
- Asking the robot open questions about what it sees. That VLM Q&A feature was dropped.

## 2. Commands to support

| Type | Examples | Needs |
|---|---|---|
| Object | "go to the printer", "take me to a trash can" | Object map |
| Named waypoint | "go to stop 3", "go to WP3", "go back to the start" | Named waypoints |
| Relations / descriptions | "the chair next to the whiteboard", "the nearest desk", "the second desk on the left" | Object map + geometry |
| Attributes | "the red chair", "the open door", "the big cardboard box" | VLM check of the object's image |
| Unseen object types | "the coffee machine" (not in the detector's 30 classes) | Open-vocabulary search of saved frames |
| Multi-step | "go to the printer, then the ladder, then come back" | All of the above, in sequence |
| Control | "stop", "cancel" | Nothing (immediate) |

## 3. Behaviour

- **Grounding:** the robot only goes to things it has **actually seen** (the detector's object
  map or saved frames) or to **named waypoints**. If nothing matches, it says so
  ("I haven't seen a coffee machine") and doesn't move.
- **Goal pose:** for an object, the robot stops about 1 m in front of it, on free floor, facing
  it. It never drives into the object's position.
- **Ambiguity** (e.g. 6 chairs):
  - **Dev mode:** a human picks. Viser lists the candidates and highlights them on the 3D view
    and mini-map; you click one.
  - **Demo mode:** the robot picks the **highest-confidence** match automatically and says which
    one it chose.
- **Before driving**, Viser shows:
  - how the command was interpreted ("→ printer #14, confidence 0.82"),
  - the target highlighted in 3D and on the mini-map,
  - which model tier answered.
- **While driving**, Viser shows status per step. "stop" cancels immediately.
- **Safety is unchanged:**
  - Every goal goes through the existing mission server, Nav2, the keep-away costmaps, and
    commander arming.
  - The language layer only picks goal poses; it never commands velocity.

## 4. Models (tiered, all on the laptop for the demo)

| Tier | Handles | Target latency |
|---|---|---|
| 0: exact match (plain code) | "go to the printer", "go to stop 3", "stop" | < 50 ms |
| 1: small local LLM | Rephrasing, relations, multi-step → structured query | < 1.5 s |
| 2: local VLM | Attribute checks on candidate images; fallback when tier 1's output doesn't validate | < 1 s per candidate |

- Each tier outputs the **same structured query**, and plain code validates it against the map
  before anything moves.
- If one tier can't handle a command, it escalates to the next tier.
- The backend is **pluggable**. A cloud model can be dropped in later, but is off by default.

## 5. Production notes (not built now)

- Commands will probably come from an edge device. The core takes text in and returns a plan,
  so it doesn't depend on Viser.
- Models must run **on the robot** in production. Tiers 0–1 should fit a Jetson Orin; the tier 2
  VLM is the risk (it may be slow on a Jetson).

## 6. Acceptance criteria (draft)

- **Test set:** ≥ 20 fixed commands covering every type in §2, run against a saved object map.
  - **Correct target:** ≥ 90 %.
  - **Wrong target** (the robot drives somewhere wrong without saying so): **0**.
  - **Unresolvable commands** (no match, or no safe goal pose) are reported, never guessed.
- **Latency** within the per-tier targets above, measured on the laptop with the demo running.
- **Live demo:** 5 commands in a row (object, waypoint, relation, attribute, multi-step) all
  arrive. The robot stays within the lap's safety envelope (0 falls, 0 contacts).
- The demo's normal lap still completes 13/13 with the feature enabled.

## 7. Open questions (fill in)

- [ ] Which local models? The proposal is in the design doc, §6.
- [ ] "Left/right": relative to the robot's current heading, or to a fixed map direction?
- [ ] When an attribute can't be confirmed from saved images, should the robot drive closer and
      look again?
- [ ] "Come back" / "go back": back to the start (spawn), or to where it was when the command was
      given?
- [ ] Waypoint names: keep "stop 1–13" for the tour, add room names ("kitchen", "printer area"),
      and/or use the upstairs route names (WP1…) once that map is aligned?
- [ ] Should the robot say anything when it arrives (e.g. a Viser toast, a camera snapshot of the
      target)?
- [ ] _add yours_
