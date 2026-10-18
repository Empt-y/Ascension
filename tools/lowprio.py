"""Run a command at idle CPU priority so builds don't steal frames from a game: python tools/lowprio.py cargo test"""
import subprocess, sys

IDLE_PRIORITY_CLASS = 0x00000040
sys.exit(subprocess.call(sys.argv[1:], creationflags=IDLE_PRIORITY_CLASS))
