"""Parse Markdown into a serializable syntax tree and structural outline."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _line_map(node: Any) -> dict[str, int] | None:
    """Convert markdown-it's zero-based half-open map to one-based inclusive lines."""
    source_map = node.map
    if source_map is None:
        return None
    return {"start": source_map[0] + 1, "end": source_map[1]}


def _node_to_dict(node: Any) -> dict[str, Any]:
    """Serialize a SyntaxTreeNode recursively without renderer normalization."""
    if node.is_root:
        return {
            "type": "root",
            "tag": "",
            "content": "",
            "attrs": {},
            "markup": "",
            "info": "",
            "line_map": None,
            "children": [_node_to_dict(child) for child in node.children],
        }

    return {
        "type": node.type,
        "tag": node.tag,
        "content": node.content,
        "attrs": dict(node.attrs),
        "markup": node.markup,
        "info": node.info,
        "line_map": _line_map(node),
        "children": [_node_to_dict(child) for child in node.children],
    }


def _plain_text(node: Any) -> str:
    """Collect readable leaf content from a syntax-tree subtree."""
    if node.children:
        return "".join(_plain_text(child) for child in node.children)
    if node.type in {"softbreak", "hardbreak"}:
        return " "
    return node.content


def _build_outline(root: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a heading hierarchy and deterministic structural warnings."""
    outline: list[dict[str, Any]] = []
    warnings: list[str] = []
    heading_path: list[str] = []
    previous_level: int | None = None

    for node in root.walk(include_self=False):
        if node.type != "heading":
            continue

        level = int(node.tag[1:])
        title = _plain_text(node).strip()
        source_map = _line_map(node)
        line_start = source_map["start"] if source_map else None
        line_end = source_map["end"] if source_map else None

        if not title:
            warnings.append(f"Empty H{level} heading at line {line_start}.")
            title = f"Untitled heading at line {line_start}"

        if previous_level is not None and level > previous_level + 1:
            warnings.append(
                f"Heading level jumps from H{previous_level} to H{level} at line {line_start}."
            )

        heading_path = heading_path[: level - 1]
        heading_path.append(title)
        outline.append(
            {
                "level": level,
                "title": title,
                "line_start": line_start,
                "line_end": line_end,
                "path": list(heading_path),
            }
        )
        previous_level = level

    if not outline:
        warnings.append(
            "No Markdown headings found; structure analysis will use the raw-text fallback."
        )

    return outline, warnings


def parse_markdown(text: str) -> dict[str, Any]:
    """Parse Markdown into a JSON-safe AST, outline, warnings, and parser metadata."""
    from markdown_it import MarkdownIt
    from markdown_it.tree import SyntaxTreeNode

    parser = MarkdownIt("commonmark").enable("table")
    root = SyntaxTreeNode(parser.parse(text))
    outline, warnings = _build_outline(root)

    try:
        parser_version = version("markdown-it-py")
    except PackageNotFoundError:
        parser_version = "unknown"

    return {
        "parser": "markdown-it-py",
        "parser_version": parser_version,
        "outline": outline,
        "warnings": warnings,
        "ast": _node_to_dict(root),
    }
