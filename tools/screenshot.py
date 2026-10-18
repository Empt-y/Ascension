"""Capture the Ascension window's client area: python tools/screenshot.py out.png"""
import ctypes, ctypes.wintypes as wt, sys
from PIL import ImageGrab

ctypes.windll.shcore.SetProcessDpiAwareness(2)
user32 = ctypes.windll.user32
hwnd = user32.FindWindowW(None, "Ascension")
if not hwnd:
    sys.exit("no Ascension window")
# The capture is a screen region, so if something else is in front (the user went back to a game)
# it would grab that instead. Refuse.
if user32.GetForegroundWindow() != hwnd:
    sys.exit("Ascension isn't the foreground window; not capturing")
rect = wt.RECT()
user32.GetClientRect(hwnd, ctypes.byref(rect))
pt = wt.POINT(0, 0)
user32.ClientToScreen(hwnd, ctypes.byref(pt))
box = (pt.x, pt.y, pt.x + rect.right, pt.y + rect.bottom)
ImageGrab.grab(bbox=box, all_screens=True).save(sys.argv[1])
print("saved", sys.argv[1], box)
