#!/usr/bin/env python3
"""Audit a SKILL.md against Codex project-local skill rules."""

from __future__ import annotations

import re
import sys
from pathlib import Path


RECOGNIZED_KEYS = {"name", "description"}


def parse_frontmatter(text: str) -> tuple[str | None, str | None]:
    if not text.startswith("---"):
        return None, None
    end = text.find("\n---", 3)
    if end == -1:
        return None, None
    return text[3:end].lstrip("\n"), text[end + 4 :]


def get_scalar(fm: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.*)$", fm, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'")


def top_level_keys(fm: str) -> list[str]:
    return [
        match.group(1)
        for line in fm.splitlines()
        if (match := re.match(r"^([A-Za-z][\w-]*):", line))
    ]


def audit(path: str) -> tuple[list[str], list[str]]:
    text = Path(path).read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    errors: list[str] = []
    warnings: list[str] = []
    if fm is None:
        return ["no valid YAML frontmatter block"], []

    name = get_scalar(fm, "name")
    if not name:
        errors.append("name: missing")
    elif not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append(
            f"name: {name!r} must use lowercase letters, digits, and hyphens only"
        )
    elif len(name) > 64:
        errors.append(f"name: {len(name)} chars > 64")

    desc = get_scalar(fm, "description")
    if not desc:
        errors.append("description: missing")
    elif "<" in desc or ">" in desc:
        errors.append("description: cannot contain angle brackets")
    elif len(desc) > 1024:
        errors.append(f"description: {len(desc)} chars > 1024")

    unknown = [key for key in top_level_keys(fm) if key not in RECOGNIZED_KEYS]
    if unknown:
        errors.append(f"unsupported Codex frontmatter keys: {', '.join(unknown)}")

    if len(body.splitlines()) > 500:
        warnings.append(
            "body is over 500 lines; consider moving detail into references"
        )

    return errors, warnings


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "SKILL.md"
    errors, warnings = audit(target)
    for warning in warnings:
        print(f"WARN  {warning}")
    for error in errors:
        print(f"ERROR {error}")
    if errors:
        print(f"FAIL {target}: {len(errors)} error(s), {len(warnings)} warning(s)")
        raise SystemExit(1)
    print(f"OK {target}: no Codex-breaking issues ({len(warnings)} warning(s))")


if __name__ == "__main__":
    main()
