# ROS2_Nav2_Mako

Autonomy stack for the Deep Robotics Lynx M20 Pro, a wheel-legged quadruped for site inspection:
ROS 2 Humble, Nav2, a commander with arming and failsafes, and natural-language navigation
("go to the ladder") that runs on three simulators and is built to move to the robot unchanged.

**Start here:** [`docs/ONBOARDING.md`](docs/ONBOARDING.md) for how the project is organised and how to
work in it, then [`CLAUDE.md`](CLAUDE.md) for the checks every change has to pass.

| Document | For |
|---|---|
| [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md) | every command: bringup, driving, maps, gates, videos, troubleshooting |
| [`docs/NL_NAVIGATION.md`](docs/NL_NAVIGATION.md), [`docs/NL_NAVIGATION_DESIGN.md`](docs/NL_NAVIGATION_DESIGN.md) | natural-language navigation: requirements and the design that was built |
| [`docs/SPLAT_DEMO.md`](docs/SPLAT_DEMO.md) | the photoreal splat simulation write-up and results |
| [`docs/NAV2_ARCHITECTURE.md`](docs/NAV2_ARCHITECTURE.md) | how the navigation stack fits together |
| [`docs/ENGINEERING_RULES.md`](docs/ENGINEERING_RULES.md) | the four binding rules for code in this repo |
| [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) | every hard-won lesson: symptom, cause, fix, rule |
| [`docs/ROADMAP.md`](docs/ROADMAP.md), [`docs/SECOND_BRAIN.md`](docs/SECOND_BRAIN.md) | milestones; ADRs, contracts, phases, changelog |

**ROS 2 distro: Humble / Ubuntu 22.04** (ADR-007). The robot's onboard system runs Foxy, but that stays
behind the vendor boundary: our stack talks to it through `m20_locomotion_bridge`, never by joining the
robot's DDS graph (ADR-014/015).

---

## What is here

**The robot stack** (`src/m20_*`): one Docker image, `m20_autonomy:latest`, built by CI and run the same way
in simulation and on the robot (ADR-011). Commander (modes, arming, failsafes, control-authority leases),
locomotion bridge (the only gateway to the vendor SDK), Nav2 configuration and behavior trees, a mission server
with a `RunMission` action, a behavior engine, and the operator bridge (Foxglove on **8765**, rosbridge on
**9090**). DDS runs on `ROS_DOMAIN_ID=42` with host networking.

**Three simulators**, all publishing the same topics and frames so the stack does not change between them:

| Simulator | Where | Used for |
|---|---|---|
| Godot splat sim (`tools/godot`) | a Godot build of a real site scan, native Windows, driven over a local JSON API | the main development and demo platform: AMCL on a site map, the object map, every NL navigation mode |
| Gazebo Harmonic (`tools/gazebo`) | gz-sim in WSL, generated M20 model, indoor rooms world | the standard-simulator counterpart; the same launchers with `RTB_SIM=gazebo` |
| MuJoCo + Gaussian splat (`tools/splat`, `tools/mujoco_sim.py`) | MuJoCo physics inside a photoreal splat of an office, Viser in the browser | the first photoreal demo and the semantic mapping tour |

**Natural-language navigation**, two phases that share the detector, the mission server and the behaviors:

- **Offline phase** (`src/m20_nlnav`): a mapping tour builds a semantic object map of the site once
  (`tools/semmap`); at run time a local LLM (Ollama) resolves the instruction against that map and Nav2 drives
  there. Handles objects, named waypoints, relations ("the chair next to the whiteboard"), attributes and
  multi-step routes. Evaluated by `tools/nlnav_eval`.
- **Real-time B** (`src/m20_vlmnav`, `tools/realtime_b`): no maps at all. Either a vision-language model picks
  the next short move from the camera (`vlm` mode), or a VLFM-style search ranks exploration frontiers with a
  value map scored by an image-text model, and a live open-vocabulary detector ends the search and starts the
  approach (`search` mode). Typed instructions ("go forward 5 m, then turn left") are parsed into exact legs.

**Demo view** (`tools/demo`): the robot's chase camera with the object boxes, the plan and the instruction's
progress drawn over it, next to the semantic map, recorded to MP4.

---

## Running it (Windows + WSL2)

The ROS containers run in Docker inside WSL2. The Godot sim runs on Windows; WSL mirrored networking lets the
containers reach it. Run the launchers from a WSL shell in the repository root; each stays in the foreground
and tears the stack down on Ctrl-C.

```bash
# Godot sim + commander + Nav2 (AMCL on the site map) + mission server + offline NL navigation
bash tools/godot/godot_sim_up.sh
tools/dev/m20 say "go to the ladder"            # natural language via the mission server
tools/dev/m20 goto 3.0 -1.5                      # one map-frame goal
tools/dev/m20 status                             # mode / failsafe / behavior snapshot

# Real-time B on the same sim (no maps): VLM moves, or the frontier search
bash tools/realtime_b/realtime_b_up.sh --mode vlm
tools/dev/m20 vlm "go to the ladder"
bash tools/realtime_b/realtime_b_up.sh --mode search
tools/dev/m20 search "refrigerator"

# Gazebo instead of Godot
docker build -f docker/Dockerfile.gazebo -t m20_gazebo:latest docker        # once
bash tools/gazebo/gazebo_sim_up.sh                                          # or RTB_SIM=gazebo with the launcher above

# the demo view (launcher in WSL, viewer on Windows)
bash tools/demo/demo_up.sh
python tools\demo\nlnav_demo_viewer.py "go to the ladder" --record ladder

# when done: stop the containers and the sim
bash tools/godot/godot_sim_down.sh
```

The commander refuses missions unless it is armed in ASSISTED or AUTONOMOUS (ADR-005); the launchers arm it
and verify the mode on `/m20/mode` before reporting ready. `/m20/mode` is latched, so a subscriber needs
transient-local QoS. The `ros2` CLI daemon inside the commander container can wedge; `tools/ros2.sh` runs a
command in a fresh container, and ROS state is best read with a small rclpy script.

**Operator view:** Foxglove Studio, "Open connection", `ws://localhost:8765`.

---

## Tests and CI

```bash
make unit    # ROS-free logic tests (pytest + rclpy shims), seconds, no Docker; CI runs these first
make image   # build the stack image
make test    # colcon test inside the image (integration)
```

Every push and pull request runs `.github/workflows/ci.yaml`: `static` (bash -n, ShellCheck, Python compile),
`unit`, and `build-and-test`; `publish-image` is a manual dispatch to GHCR. Keep node logic testable at the
unit level: side effects behind injectable objects, state machines as plain methods.

## How changes land

The default branch is `3dgs-sim-nl-navigation-offline`. It is protected: changes arrive by pull request, the
three CI jobs must pass on an up-to-date branch, and review conversations must be resolved.

1. Branch from the default branch, make the change with its test, run `make unit`.
2. Run `/code-review` in Claude Code before pushing; fix what it finds.
3. Open the PR with the template (summary with the measured reason, test plan, review notes).
4. `.github/workflows/claude-code-review.yml` runs the same review on the PR and posts its findings as
   comments; `@claude` in a comment asks for a specific follow-up. Address each Important finding or answer it.
5. Merge when CI is green and the branch is current (use "Update branch" if it fell behind).

`CLAUDE.md` holds the project rules the reviews apply; `REVIEW.md` tunes what the hosted review reports.
Commit messages give the measured reason for a change. `notes/`, `.claude/` and editor workspace files are
never committed.

---

## Layout

- `src/m20_msgs` interface contracts (designed first)
- `src/m20_locomotion_bridge` sole gateway to the Deep Robotics SDK; cmd/gait + watchdog
- `src/m20_commander` modes, arming, failsafes, control-authority leases
- `src/m20_navigation` Nav2 config + behavior trees
- `src/m20_perception` SLAM / EKF / costmap config
- `src/m20_missions` mission server (`RunMission`) + agent interface
- `src/m20_behaviors` behavior engine + plugins
- `src/m20_nlnav` offline natural-language navigation (LLM + object map)
- `src/m20_vlmnav` Real-time B: VLM moves, VLFM-style search, instruction parsing, shared behaviors
- `src/m20_station` operator bridge (foxglove_bridge + rosbridge)
- `src/m20_bringup` top-level launch
- `docker/` Dockerfiles (stack, sim, Gazebo, splat), compose, healthcheck, entrypoint
- `tools/godot`, `tools/gazebo`, `tools/splat` the three simulators and their glue nodes
- `tools/realtime_b` Real-time B launchers, model servers, eval cases
- `tools/semmap`, `tools/nlnav_eval` semantic mapping tour; NL navigation evaluation harness
- `tools/demo` the demo view and its launcher
- `tools/dev/m20` operator CLI over the mission server
- `tools/nav` bringup and teardown of the ROS stack (`bringup_plan_a.sh`, `kill_stack.sh`)
- `tests/` the unit tests `make unit` runs
