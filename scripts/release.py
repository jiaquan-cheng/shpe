import argparse
import json
import re
import subprocess
from pathlib import Path


def bump(version: str) -> None:
    """Bump the version in pyproject.toml, package.json, and launch.json."""
    tag = f"v{version}"
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(
        re.sub(r'version\s*=\s*"[^"]+"', f'version = "{version}"', content, count=1),
        encoding="utf-8",
    )

    pkg_path = Path("shpe-vscode/package.json")
    data = json.loads(pkg_path.read_text(encoding="utf-8"))
    data["version"] = version
    pkg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    launch_path = Path("shpe-vscode/.vscode/launch.json")
    data = json.loads(launch_path.read_text(encoding="utf-8"))
    data["version"] = version
    launch_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "add",
            "pyproject.toml",
            "shpe-vscode/package.json",
            "shpe-vscode/.vscode/launch.json",
        ],
        check=True,
    )
    subprocess.run(["git", "commit", "-m", f"chore: release {tag}"], check=True)
    subprocess.run(["git", "tag", tag], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()
    bump(args.version)
