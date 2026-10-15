#!/usr/bin/env bash
# Stop the Godot stack and free the machine (end of a session): the ROS stack (kill_stack.sh: Nav2, AMCL, mission
# server, bridge, NL navigation containers), the local LLM (m20_ollama), then ask the Godot sim to quit (a clean
# quit through its API, not a kill). The counterpart of godot_sim_up.sh.
#   bash tools/godot/godot_sim_down.sh          leaves docker-commander-1 up (idle, ~0 CPU; godot_sim_up.sh reuses it)
#   bash tools/godot/godot_sim_down.sh --all    also `docker compose down` (everything off; the next bringup recreates it)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ALL=0
case "${1:-}" in
  --all) ALL=1 ;;
  "") ;;
  *) echo "usage: godot_sim_down.sh [--all]"; exit 2 ;;
esac

echo "[sim_down] ROS stack (kill_stack.sh)"
bash "$REPO/tools/nav/kill_stack.sh"

if docker ps --format '{{.Names}}' | grep -qx m20_ollama; then
  echo "[sim_down] stopping the local LLM (m20_ollama; the model reloads on the next godot_sim_up.sh)"
  docker stop m20_ollama >/dev/null
fi

echo "[sim_down] asking the Godot sim to quit"
python3 - <<'PY'
import socket
try:
    s = socket.create_connection(("127.0.0.1", 7799), timeout=3)
    s.sendall(b'{"cmd":"quit"}\n')
    print("  quit sent")
except OSError as e:
    print(f"  sim API not reachable ({e}): Godot is probably not running")
PY

if [ "$ALL" = "1" ]; then
  echo "[sim_down] compose services (docker compose down)"
  docker compose -f "$REPO/docker/compose.yaml" down
fi

echo "[sim_down] containers still running:"
docker ps --format '  {{.Names}}\t{{.Status}}'
