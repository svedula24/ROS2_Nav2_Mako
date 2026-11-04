# NL navigation demo (Godot sim)

A demo view of the offline natural-language navigation ("go to the ladder"): the robot's chase view with the
semantic map drawn over it, the robot on the map, and the instruction and its progress in words. It is a
separate branch (`nlnav-demo`, this worktree) with its own launcher; the dev stack is untouched, and the Godot
sim is not modified (everything is drawn outside it).

## Run

```
# WSL, from this worktree
bash tools/demo/demo_up.sh                       # Godot + bridge + Nav2 (AMCL) + mission + NL node; waits until ready
# Windows
python tools\demo\nlnav_demo_viewer.py "go to the ladder" --record ladder
# when done
bash tools/godot/godot_sim_down.sh
```

The viewer opens one window and, with `--record NAME`, also writes `Videos\nlnav_demo\NAME.mp4` and a mid-run
still `NAME.jpg`. Keys: `q` quit (cancels the goal if it is still running), `t` show the true pose on the map
as a grey ring, `b` hide/show the boxes in the chase view. `--no-send` only watches; `--duration S` stops after
S seconds.

## What is on the screen

**Chase view (left)**, the Godot window captured while it stays parked off-screen:
- every object the navigator knows, as its 3D box from the object map, with its label and id; the object the
  navigator chose is orange and carries its score; the others are cyan (within 9 m, in front of the camera);
- Nav2's global plan as a green line on the floor and the goal pose as a green ring;
- the banner: the instruction, then what is happening now: `Listening...` -> `Understanding the instruction
  (LLM + object map)...` -> `Driving to the ladder (Nav2)  6.8 m to go` -> `Arrived at the ladder (41 s)`;
  the target's class, id and score on the right.

**Semantic map (right)**: the site map Nav2 localizes against (light = free, dark = wall, grey = unknown), every
kept object as a box with its label, the robot where the STACK believes it is (red triangle, from TF
`map -> base_link`, i.e. AMCL), its trail (blue), the plan (green), the goal (green dot and arrow), the target
(pulsing orange ring).

## How it works

- `nlnav_demo_relay.py` runs inside `docker-commander-1` (the viewer copies it in and pipes JSON lines back):
  TF pose at 10 Hz, `/plan`, the navigator's markers on `/m20/nlnav/markers` (its chosen target, goal and
  candidates, published when it resolves, before it drives), and it is the NavigateNL action client: the
  viewer tells it `send <instruction>` once it is recording, and gets feedback and the result back.
- `nlnav_demo_viewer.py` (Windows, numpy + OpenCV) captures the Godot window with `PrintWindow`
  (works off-screen, costs the sim nothing) and places the spectator camera itself every tick
  (`{"cmd":"viewpoint"}`, the sim's own chase offsets and lags from `spectator.gd`), so it knows the camera pose
  exactly and projects the object boxes with Godot's default 75° vertical FOV. The sim's `screenshot` command is
  never used: it stalls the sim's main thread and corrupts odometry.
- Objects shown = the object map entries with `label_score >= 0.4`, the navigator's own cut (below it the
  navigator keeps the object but calls it "object"). The object map is `~/m20_sim/semmap/circuit4/object_map.json`
  in WSL (built by `tools/semmap`), the site map `tools/sim/godot_circuit4_map.*`.

## Gotchas

- The Godot window must exist and not be minimized (parked off-screen is fine). `demo_up.sh` parks it.
- Record with the viewer, not with a screen recorder: the composed frame is what you want, and the window is
  off-screen anyway.
- If the robot is lost (AMCL far from the truth: press `t` to compare), reset: sim `{"cmd":"reset"}` plus an
  `/initialpose` at the spawn (`tools/nlnav_eval/run_eval.py` does this between cases).
