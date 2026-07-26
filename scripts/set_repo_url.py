"""Write a GitHub repo URL into the bootstrap cell of every Colab notebook.

Each Colab notebook clones this repository as its first step, so each one needs
the URL. Editing eight files by hand is tedious and easy to get half-right -- a
notebook still pointing at CHANGEME fails several cells in, with a confusing
error about a missing module.

Invoked by scripts/set_repo_url.ps1, but runnable directly:

    python scripts/set_repo_url.py https://github.com/user/repo.git

This lives in a real file rather than inside a PowerShell here-string passed to
`python -c`. That approach cannot survive here: PowerShell strips bare double
quotes when building the argument list for a native executable, so any Python
source containing string literals arrives corrupted.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "notebooks" / "colab"


def normalize(url: str) -> str:
    """Accept an SSH remote or a browser URL; always emit the https clone form.

    Colab clones anonymously and has no SSH key, so an SSH remote would fail
    there with a permission error that looks nothing like its actual cause.
    """
    url = url.strip().rstrip("/")

    m = re.match(r"^git@github\.com:(.+?)(?:\.git)?$", url)
    if m:
        url = f"https://github.com/{m.group(1)}"
        print("  converted SSH remote to https (Colab has no SSH key)")

    if not url.endswith(".git"):
        url += ".git"

    if "CHANGEME" in url:
        raise SystemExit("error: that URL still contains CHANGEME")
    if not url.startswith("https://"):
        raise SystemExit(f"error: expected an https URL, got {url!r}")

    return url


def apply(url: str) -> int:
    changed = 0
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"error: no notebooks found in {NB_DIR}")

    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        hit = False

        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            for i, line in enumerate(cell["source"]):
                if line.lstrip().startswith("REPO_URL = "):
                    # json.dumps supplies the surrounding quotes and escaping.
                    new = "REPO_URL = " + json.dumps(url) + "\n"
                    if line != new:
                        cell["source"][i] = new
                        hit = True

        if hit:
            nb_path.write_text(
                json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
            )
            changed += 1
            print(f"  updated  {nb_path.name}")
        else:
            print(f"  ok       {nb_path.name}")

    return changed


def verify(url: str) -> None:
    """Re-read every notebook and confirm the line is valid Python.

    This is the check that would have caught the quoting bug that produced
    `REPO_URL = https://...` with no quotes -- valid JSON, invalid Python.
    """
    import ast

    bad = []
    for nb_path in sorted(NB_DIR.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        found = False
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            for line in cell["source"]:
                if line.lstrip().startswith("REPO_URL = "):
                    found = True
                    try:
                        tree = ast.parse(line.strip())
                        value = tree.body[0].value  # type: ignore[attr-defined]
                        if not isinstance(value, ast.Constant) or value.value != url:
                            bad.append(f"{nb_path.name}: unexpected value {line.strip()!r}")
                    except SyntaxError:
                        bad.append(f"{nb_path.name}: not valid Python -> {line.strip()!r}")
        if not found:
            bad.append(f"{nb_path.name}: no REPO_URL line found")

    if bad:
        print("\nVERIFICATION FAILED:")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    print("  verified: every REPO_URL line parses and matches")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/set_repo_url.py <repo-url>")

    url = normalize(sys.argv[1])
    print(f"  {url}")
    apply(url)
    verify(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
