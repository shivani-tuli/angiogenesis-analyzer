"""
Build script — creates a standalone .app (macOS) or .exe (Windows)
using PyInstaller.
"""
import subprocess
import sys
import platform

APP_NAME = "Angiogenesis Analyzer"
ENTRY = "app.py"
ICON = None  # set to an .icns / .ico path if desired

def build():
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--windowed",           # no console window
        "--onedir",             # directory bundle (required for macOS .app)
        f"--name={APP_NAME}",
        "--hidden-import=skimage",
        "--hidden-import=skimage.morphology",
        "--hidden-import=skimage.filters",
        "--hidden-import=skan",
        "--hidden-import=PIL",
        "--hidden-import=openpyxl",
        "--collect-submodules=skimage",
        "--collect-submodules=skan",
    ]
    if ICON:
        cmd.append(f"--icon={ICON}")
    cmd.append(ENTRY)

    print(f"Building {APP_NAME} for {platform.system()}…")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"\n✅  Done!  Find your app in the  dist/  folder.")

if __name__ == "__main__":
    build()
