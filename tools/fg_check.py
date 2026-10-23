"""Exit 1 if the foreground window belongs to anything other than an allow-listed app.

Live tests open windows and steal focus; never do that while the user is in a game.
"""
import ctypes, ctypes.wintypes as wt, os, sys

# prismlauncher.exe, discord.exe, chrome.exe: fine to take focus from (user, 2026-10-18); games are not.
# spotify.exe, javaw.exe (GregTech: New Horizons, Minecraft): fine too (user, 2026-10-23: "I'll tell you not to open if I'm busy").
ALLOWED = {"claude.exe", "explorer.exe", "ascension.exe", "windowsterminal.exe", "code.exe", "fsh-shell.exe", "prismlauncher.exe", "discord.exe", "chrome.exe", "spotify.exe", "javaw.exe"}

user32, kernel32, psapi = ctypes.windll.user32, ctypes.windll.kernel32, ctypes.windll.psapi
hwnd = user32.GetForegroundWindow()
pid = wt.DWORD()
user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
h = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
buf = ctypes.create_unicode_buffer(1024)
size = wt.DWORD(1024)
name = ""
if h and kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
    name = os.path.basename(buf.value).lower()
print(f"foreground: {name or '?'}")
# A game running at all blocks launches, even with something else in front for a moment (the user
# flicks to Spotify or Discord mid-match).
GAMES = {"overwatch.exe", "arkascended.exe", "shootergame.exe"}
import subprocess
running = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout.lower()
playing = sorted(g for g in GAMES if f'"{g}"' in running)
if playing:
    print(f"game running: {', '.join(playing)}")
    sys.exit(1)
sys.exit(0 if name in ALLOWED else 1)
