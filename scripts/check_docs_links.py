#!/usr/bin/env python3
"""Check Markdown links and heading anchors in the repository docs.

Scans every Markdown file under ``docs/``, ``examples/`` and the repository
root, extracts inline ``[text](target)`` links, and verifies that

* relative links resolve to an existing file, and
* ``#anchor`` fragments match a heading in the target file.

External (``http(s)://``), ``mailto:`` and pure-fragment links are handled
sensibly: external URLs are skipped, and a bare ``#anchor`` is checked against
the file it appears in.

Usage::

    python scripts/check_docs_links.py [paths ...]

Exits non-zero if any link is broken.  No third-party dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: ``[text](target)`` — target may not contain unescaped whitespace or parens.
LINK_RE = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+?)(?:\s+\"[^\"]*\")?\s*\)")
#: ATX heading, e.g. ``## Some Title``.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def slugify(text: str) -> str:
    """Approximate GitHub's heading -> anchor slug algorithm."""
    text = text.strip().lower()
    # Drop inline code/emphasis markers and other punctuation.
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def anchors_for(path: Path) -> set[str]:
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADING_RE.match(line)
        if not m:
            continue
        slug = slugify(m.group(2))
        # GitHub de-duplicates repeated headings with ``-1``, ``-2``, ...
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}-{seen[slug]}"
        else:
            seen[slug] = 0
        anchors.add(slug)
    return anchors


def markdown_files(paths: list[str]) -> list[Path]:
    if paths:
        files: list[Path] = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                files.extend(sorted(path.rglob("*.md")))
            else:
                files.append(path)
        return files
    found = sorted((ROOT / "docs").rglob("*.md"))
    found += sorted((ROOT / "examples").rglob("*.md"))
    found += [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "FINAL_REPORT.md"]
    return [f for f in found if f.exists()]


def check_file(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    cache: dict[Path, set[str]] = {}

    def anchors(target: Path) -> set[str]:
        if target not in cache:
            cache[target] = anchors_for(target)
        return cache[target]

    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in LINK_RE.finditer(line):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, fragment = target.partition("#")
            if file_part:
                resolved = (path.parent / file_part).resolve()
                if not resolved.exists():
                    errors.append(f"{path.relative_to(ROOT)}:{lineno}: broken link -> {target}")
                    continue
            else:
                resolved = path
            if fragment and resolved.suffix == ".md" and fragment not in anchors(resolved):
                errors.append(
                    f"{path.relative_to(ROOT)}:{lineno}: missing anchor #{fragment} in {file_part or path.name}"
                )


def main(argv: list[str]) -> int:
    files = markdown_files(argv)
    errors: list[str] = []
    for path in files:
        check_file(path, errors)
    if errors:
        print(f"Found {len(errors)} broken link(s):")
        for err in errors:
            print(f"  {err}")
        return 1
    print(f"OK: checked {len(files)} Markdown file(s), no broken links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
