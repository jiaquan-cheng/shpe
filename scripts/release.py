import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bump(version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Version must use MAJOR.MINOR.PATCH format")

    tag = f"v{version}"

    pyproject = ROOT / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    content, project_matches = re.subn(
        r'(?m)^version\s*=\s*"[^"]+"', f'version = "{version}"', content, count=1
    )

    gradle_properties = ROOT / "editors/pycharm/gradle.properties"
    gradle_content = gradle_properties.read_text(encoding="utf-8")
    gradle_content, gradle_matches = re.subn(
        r"(?m)^version\s*=\s*[^\n]+", f"version = {version}", gradle_content, count=1
    )
    if project_matches != 1 or gradle_matches != 1:
        raise ValueError("Could not find project and PyCharm versions to update")

    pyproject.write_text(content, encoding="utf-8")

    for path in (
        ROOT / "editors/vscode/package.json",
        ROOT / "editors/vscode/package-lock.json",
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = version
        if "packages" in data:
            data["packages"][""]["version"] = version
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    launch = ROOT / "editors/vscode/.vscode/launch.json"
    data = json.loads(launch.read_text(encoding="utf-8"))
    data["version"] = version
    launch.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    gradle_properties.write_text(gradle_content, encoding="utf-8")

    paths = [
        "pyproject.toml",
        "editors/vscode/package.json",
        "editors/vscode/package-lock.json",
        "editors/vscode/.vscode/launch.json",
        "editors/pycharm/gradle.properties",
    ]
    subprocess.run(["git", "add", *paths], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore: release {tag}", "--", *paths],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "tag", tag], cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    bump(parser.parse_args().version)
