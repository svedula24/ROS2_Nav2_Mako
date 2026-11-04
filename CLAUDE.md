# ROS2_Nav2_Mako

Autonomy stack for the M20 robot: ROS 2 Humble, Nav2, a commander with failsafes, natural-language navigation
(offline object map, and the Real-time B camera-driven search), on the Godot splat sim, Gazebo, or the robot.

## Run the checks before you claim something works

- `make unit` (or `python3 -m pytest tests -q`): the ROS-free logic tests with rclpy shims, seconds, no Docker.
  CI runs them first; every behavior change ships with a test here.
- `bash -n` and `shellcheck -S error` on every shell script, `python3 -m compileall -q src tools tests`: CI's
  `static` job. Node code that needs real ROS is tested by `colcon test` inside the stack image (CI's
  `build-and-test`).
- Live checks run on the Godot stack (`tools/godot/godot_sim_up.sh`, `tools/demo/demo_up.sh`) or Gazebo
  (`tools/gazebo/gazebo_sim_up.sh`). Shut the stack down afterwards (`tools/godot/godot_sim_down.sh`).

## Rules that bind every script and node (docs/ENGINEERING_RULES.md)

1. No silent failure paths: `set -euo pipefail`, no `|| true` or `2>/dev/null` on anything that can fail
   meaningfully, a check that cannot find its input fails with a message.
2. Verify the contract before writing the consumer: message type, publisher QoS (`/m20/mode` is latched),
   one real sample of any sensor data.
3. Smoke-test every script on tiny input before wiring it into a chain.
4. Two identical environmental failures mean redesign, not a third retry.

## What a review should flag first

- Motion safety: a stop, cancel, crash or timeout path that does not leave the robot at zero velocity; a goal
  sent while the previous one is still being aborted; a watchdog or stuck rule whose condition cannot become
  true; a failsafe or arm check bypassed or reported as passed without reading `/m20/mode`.
- Launchers: background work not stopped on failure paths; success printed after a refused step; checks that go
  through the `ros2` CLI daemon (it wedges in the commander container: read ROS state with a small rclpy script).
- Threads: rclpy callbacks and reader threads share state with the main loop only under a lock; a deque or
  list must not be iterated while another thread appends.
- Instruction parsing (`src/m20_vlmnav/m20_vlmnav/instruction.py`, `targets.py`, `moves.py`): a typed number
  means what it says (split or refuse, never silently clamp); every regex change comes with the phrasing it was
  made for as a test.
- Duplication across the Godot, Gazebo and demo tools: the shared helpers live in `tools/godot/godot_io.py`
  and `src/m20_vlmnav/m20_vlmnav/behaviors.py`; a second copy is a finding.

## Conventions

- Commit messages explain the measured reason for a change (what failed, what number was seen), not only what
  changed. Each retrofit to the engineering rules is named in its commit message.
- `notes/` is never committed. Recorded runs go to `~/m20_sim/...`, not the repository.
- Memory and session files (`.claude/`, editor workspace files) stay out of commits.
