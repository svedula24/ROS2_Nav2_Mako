# Godot sim window placement.  powershell.exe -NoProfile -ExecutionPolicy Bypass -File godot_window.ps1 park|show|state
#   park   move the window almost entirely off-screen (a 40x40 px corner stays on the desktop) so the desk is
#          free for other work. Godot keeps rendering an off-screen window; it stops only when MINIMIZED, and
#          a minimized or closed window means no camera/lidar frames (the bridge's streams time out).
#   show   restore + move on-screen (100,60; 1600x900), for watching or recording a run.
#   state  print position, size and whether it is minimized.
param([ValidateSet("park", "show", "state")] [string] $Mode = "state")
Add-Type @"
using System; using System.Runtime.InteropServices;
public struct GRECT { public int L, T, R, B; }
public class GW {
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int hh, bool r);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out GRECT r);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
}
"@
[void][GW]::SetProcessDPIAware()
$p = $null
for ($i = 0; $i -lt 30 -and -not $p; $i++) {          # the window appears a few seconds after the process
  # by title: the Godot editor is the same exe, and must not be the window that gets parked
  $p = Get-Process Godot* -ErrorAction SilentlyContinue |
       Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like "Bright Robotics Simulator*" } |
       Select-Object -First 1
  if (-not $p) { Start-Sleep 1 }
}
if (-not $p) { Write-Output "godot_window: no Godot window"; exit 1 }
$h = $p.MainWindowHandle
Add-Type -AssemblyName System.Windows.Forms
# the whole desktop across monitors (physical px, since we're DPI-aware): parking past the PRIMARY screen's edge
# would land the window on a monitor docked to its right
$wa = [System.Windows.Forms.SystemInformation]::VirtualScreen
switch ($Mode) {
  "park" { [void][GW]::ShowWindowAsync($h, 9); [void][GW]::MoveWindow($h, $wa.Right - 40, $wa.Bottom - 40, 1600, 900, $true) }
  "show" { [void][GW]::ShowWindowAsync($h, 9); [void][GW]::MoveWindow($h, 100, 60, 1600, 900, $true) }
}
Start-Sleep -Milliseconds 300
$r = New-Object GRECT; [void][GW]::GetWindowRect($h, [ref]$r)
Write-Output ("godot_window: pid {0} at {1},{2} size {3}x{4} minimized={5}" -f $p.Id, $r.L, $r.T, ($r.R - $r.L), ($r.B - $r.T), [GW]::IsIconic($h))
