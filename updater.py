"""Auto-updater for RunnerBibTagger using GitHub Releases."""

import json
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from version import __version__, APP_NAME, GITHUB_REPO


@dataclass
class UpdateInfo:
    """Information about an available update."""
    version: str
    download_url: str
    release_notes: str
    file_size: int


def parse_version(version_str: str) -> tuple:
    """Parse version string into comparable tuple."""
    # Handle versions like "1.0.0", "1.2.3-beta", etc.
    clean = version_str.split("-")[0]  # Remove pre-release suffix
    parts = clean.split(".")
    return tuple(int(p) for p in parts if p.isdigit())


def get_platform_asset_name() -> str:
    """Get the expected asset name for the current platform."""
    system = platform.system()
    if system == "Windows":
        return f"{APP_NAME}-Windows.exe"
    elif system == "Darwin":
        return f"{APP_NAME}-macOS.zip"
    elif system == "Linux":
        return f"{APP_NAME}-Linux"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def check_for_updates() -> Optional[UpdateInfo]:
    """
    Check GitHub Releases for a newer version.

    Returns UpdateInfo if an update is available, None otherwise.
    Raises exception on network errors.
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github.v3+json", "User-Agent": APP_NAME}
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            release = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # No releases yet
        raise
    except urllib.error.URLError:
        return None  # Network error, fail silently

    latest_version = release["tag_name"].lstrip("v")
    current = parse_version(__version__)
    latest = parse_version(latest_version)

    if latest <= current:
        return None  # Already up to date

    # Find the asset for this platform
    asset_name = get_platform_asset_name()
    download_url = None
    file_size = 0

    for asset in release.get("assets", []):
        if asset["name"] == asset_name:
            download_url = asset["browser_download_url"]
            file_size = asset["size"]
            break

    if not download_url:
        return None  # No asset for this platform

    return UpdateInfo(
        version=latest_version,
        download_url=download_url,
        release_notes=release.get("body", ""),
        file_size=file_size
    )


def download_update(
    update_info: UpdateInfo,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    """
    Download the update to a temporary location.

    Args:
        update_info: The update to download
        progress_callback: Optional callback(bytes_downloaded, total_bytes)

    Returns:
        Path to the downloaded file
    """
    asset_name = get_platform_asset_name()
    temp_dir = Path(tempfile.gettempdir())
    download_path = temp_dir / f"{APP_NAME}-update-{asset_name}"

    request = urllib.request.Request(
        update_info.download_url,
        headers={"User-Agent": APP_NAME}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        total_size = update_info.file_size or int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024  # 64KB chunks

        with open(download_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total_size)

    return download_path


def get_current_executable() -> Path:
    """Get the path to the current executable."""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        return Path(sys.executable)
    else:
        # Running as script (for development)
        return Path(__file__).parent / "main.py"


def apply_update(downloaded_path: Path) -> None:
    """
    Apply the update by replacing the current executable.

    This generates a platform-specific script that:
    1. Waits for the current process to exit
    2. Replaces the executable
    3. Restarts the application
    4. Deletes itself
    """
    current_exe = get_current_executable()
    system = platform.system()

    if system == "Windows":
        _apply_update_windows(downloaded_path, current_exe)
    elif system == "Darwin":
        _apply_update_macos(downloaded_path, current_exe)
    elif system == "Linux":
        _apply_update_linux(downloaded_path, current_exe)
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def _apply_update_windows(downloaded_path: Path, current_exe: Path) -> None:
    """Apply update on Windows using a batch script."""
    script_path = Path(tempfile.gettempdir()) / f"{APP_NAME}-update.bat"

    # Get the directory where the exe lives
    install_dir = current_exe.parent
    exe_name = current_exe.name

    script_content = f'''@echo off
:: Wait for the application to close
timeout /t 2 /nobreak > nul

:: Replace the executable
move /y "{downloaded_path}" "{install_dir / exe_name}"

:: Restart the application
start "" "{install_dir / exe_name}"

:: Delete this script
del "%~f0"
'''

    with open(script_path, "w") as f:
        f.write(script_content)

    # Run the script detached from this process
    subprocess.Popen(
        ["cmd", "/c", str(script_path)],
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True
    )

    # Exit the current application
    sys.exit(0)


def _apply_update_macos(downloaded_path: Path, current_exe: Path) -> None:
    """Apply update on macOS using a shell script."""
    script_path = Path(tempfile.gettempdir()) / f"{APP_NAME}-update.sh"

    # For macOS, the download is a zip containing the .app bundle
    # The current_exe is inside the .app bundle
    # We need to find the .app directory and replace it

    # Find the .app bundle (go up from Contents/MacOS/RunnerBibTagger)
    app_bundle = current_exe
    while app_bundle.suffix != ".app" and app_bundle.parent != app_bundle:
        app_bundle = app_bundle.parent

    if app_bundle.suffix != ".app":
        # Fallback: assume standard location
        app_bundle = current_exe.parent.parent.parent

    install_dir = app_bundle.parent
    app_name = app_bundle.name

    script_content = f'''#!/bin/bash
# Wait for the application to close
sleep 2

# Remove old app bundle
rm -rf "{install_dir / app_name}"

# Unzip new version
unzip -o "{downloaded_path}" -d "{install_dir}"

# Remove the downloaded zip
rm "{downloaded_path}"

# Restart the application
open "{install_dir / app_name}"

# Delete this script
rm "$0"
'''

    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)

    # Run the script detached
    subprocess.Popen(
        ["/bin/bash", str(script_path)],
        start_new_session=True,
        close_fds=True
    )

    sys.exit(0)


def _apply_update_linux(downloaded_path: Path, current_exe: Path) -> None:
    """Apply update on Linux using a shell script."""
    script_path = Path(tempfile.gettempdir()) / f"{APP_NAME}-update.sh"

    install_dir = current_exe.parent
    exe_name = current_exe.name

    script_content = f'''#!/bin/bash
# Wait for the application to close
sleep 2

# Make the new binary executable
chmod +x "{downloaded_path}"

# Replace the executable
mv -f "{downloaded_path}" "{install_dir / exe_name}"

# Restart the application
"{install_dir / exe_name}" &

# Delete this script
rm "$0"
'''

    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)

    # Run the script detached
    subprocess.Popen(
        ["/bin/bash", str(script_path)],
        start_new_session=True,
        close_fds=True
    )

    sys.exit(0)
