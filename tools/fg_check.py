"""Exit 1 if the foreground window belongs to anything other than an allow-listed app.

Live tests open windows and steal focus; never do that while the user is in a game.
"""
import ctypes, ctypes.wintypes as wt, os, sys

# prismlauncher.exe: the Minecraft launcher itself is fine (user, 2026-10-18); the game it starts is not.
ALLOWED = {"claude.exe", "explorer.exe", "ascension.exe", "windowsterminal.exe", "code.exe", "fsh-shell.exe", "prismlauncher.exe"}

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
sys.exit(0 if name in ALLOWED else 1)
