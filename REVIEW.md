# Review instructions

Read by Claude Code Review (the hosted service) from the default branch. The local `/code-review` skill and the
GitHub Actions review read CLAUDE.md instead, so the rules both should apply live there; this file only tunes
what a pull-request review reports and at what severity.

## What Important means here

Reserve Important for findings that would move the robot wrongly, leave it moving, or hide a failure:
a velocity command that is not zeroed on a stop, cancel or crash path; a goal sent while the previous one is
still being aborted; a watchdog, timeout or stuck rule that cannot fire; a failsafe or mode check bypassed;
a launcher that reports success after a refused or unverified step; a silent failure path (`|| true`,
`2>/dev/null`, a check that computes on empty input) on something that matters; a thread-safety slip in a
node or viewer. Style, naming, duplication and refactoring suggestions are Nit at most.

## Cap the nits

Report at most five Nits per review.

## Do not report

- Anything CI already enforces: `bash -n`, ShellCheck errors, Python compile errors, the unit tests
- Content under `notes/`, `docs/assets/`, and recorded data (`*.mp4`, `*.jpg`, `*.png`, `*.pgm`, `*.ply`, `*.jsonl`)
- Generated or vendored files: `tools/gazebo/models/**` (built by `make_m20_sdf.py`), `*.lock`
- Hard-coded tuning numbers that the commit message or the comment next to them explains with a measurement

## Always check

- A new or changed cancel, stop, crash or timeout path still leaves the robot with zero velocity, and says in a
  comment who sends that zero (the behavior server, the glue node, the commander)
- A new subscriber matches the publisher's QoS (latched topics such as `/m20/mode` are transient_local)
- A shell script that starts background work also stops it on every failure path
- A regex over an operator's instruction has a test for the phrasing it was changed for
- A fix that claims to close a race or a watchdog gap is not dead code (the condition can actually be true)

## Evidence

A claim about runtime behavior needs a `file:line` citation in the source, not an inference from a name.
