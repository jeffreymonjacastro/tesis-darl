"""Tests for Markdown AST extraction and batch artifact integration."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from extractor.parsers.markdown import parse_markdown
from extractor.utils import extract_single_file, main


def _all_nodes(node: dict) -> list[dict]:
    """Flatten a serialized AST for concise structural assertions."""
    descendants = [
        descendant for child in node["children"] for descendant in _all_nodes(child)
    ]
    return [node, *descendants]


def _write_markdown(path: Path, text: str) -> Path:
    """Write a Markdown fixture with exact UTF-8 content."""
    path.write_text(text, encoding="utf-8")
    return path


def _patch_outputs(
    monkeypatch: pytest.MonkeyPatch, output_dir: Path
) -> tuple[Path, Path, Path]:
    """Redirect all extractor artifacts to a test-local directory."""
    output_text = output_dir / "full_text.txt"
    output_meta = output_dir / "metadata.json"
    output_ast = output_dir / "markdown_ast.json"
    monkeypatch.setattr("extractor.utils.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("extractor.utils.OUTPUT_TEXT", output_text)
    monkeypatch.setattr("extractor.utils.OUTPUT_META", output_meta)
    monkeypatch.setattr("extractor.utils.OUTPUT_AST", output_ast)
    monkeypatch.setattr("extractor.utils.prepare_dependencies", lambda *args: None)
    return output_text, output_meta, output_ast


class TestMarkdownAst:
    """Verify the parser's hierarchical, source-aware JSON contract."""

    def test_ast_preserves_rich_markdown_structure(self):
        source = """# Title

Paragraph with *emphasis* and a [link](https://example.com).

- first item
- second item

| Name | Value |
| --- | --- |
| alpha | 1 |

```python
print("hello")
```
"""

        result = parse_markdown(source)
        nodes = _all_nodes(result["ast"])
        node_types = {node["type"] for node in nodes}

        assert {
            "root",
            "heading",
            "paragraph",
            "em",
            "link",
            "bullet_list",
            "table",
            "fence",
        } <= node_types
        link = next(node for node in nodes if node["type"] == "link")
        fence = next(node for node in nodes if node["type"] == "fence")
        assert link["attrs"]["href"] == "https://example.com"
        assert fence["info"] == "python"
        assert fence["content"] == 'print("hello")\n'

    def test_outline_has_inclusive_lines_hierarchy_and_jump_warning(self):
        result = parse_markdown("# Part\n\n## Chapter\n\n#### Detail\n")

        assert result["outline"] == [
            {
                "level": 1,
                "title": "Part",
                "line_start": 1,
                "line_end": 1,
                "path": ["Part"],
            },
            {
                "level": 2,
                "title": "Chapter",
                "line_start": 3,
                "line_end": 3,
                "path": ["Part", "Chapter"],
            },
            {
                "level": 4,
                "title": "Detail",
                "line_start": 5,
                "line_end": 5,
                "path": ["Part", "Chapter", "Detail"],
            },
        ]
        assert result["warnings"] == ["Heading level jumps from H2 to H4 at line 5."]

    @pytest.mark.parametrize(
        ("source", "warning_fragment"),
        [
            ("Paragraph only.\n", "No Markdown headings found"),
            ("#\n", "Empty H1 heading"),
        ],
    )
    def test_outline_warnings(self, source, warning_fragment):
        result = parse_markdown(source)
        assert any(warning_fragment in warning for warning in result["warnings"])


class TestMarkdownExtractionFallback:
    """Verify raw-text continuity when optional AST parsing cannot run."""

    def test_raw_markdown_is_preserved_after_successful_parse(self, tmp_path):
        source = "# Título\n\nTexto con  dos espacios.  \n"
        markdown_path = _write_markdown(tmp_path / "notes.md", source)

        with mock.patch("extractor.utils.prepare_dependencies"):
            result = extract_single_file(markdown_path, "text", "no")

        assert result["text"] == source
        assert result["extraction_method"] == "markdown-it-py"
        assert result["markdown_ast"] is not None

    @pytest.mark.parametrize("error", [ImportError("missing"), RuntimeError("broken")])
    def test_parser_failure_falls_back_without_aborting(self, tmp_path, error):
        source = "# Still available\n\nRaw text.\n"
        markdown_path = _write_markdown(tmp_path / "fallback.md", source)

        with (
            mock.patch("extractor.utils.prepare_dependencies"),
            mock.patch("extractor.utils.parse_markdown", side_effect=error),
        ):
            result = extract_single_file(markdown_path, "text", "no")

        assert result["text"] == source
        assert result["extraction_method"] == "plain-text"
        assert result["markdown_ast"] is None
        assert "raw Markdown text was preserved" in result["warnings"][0]


class TestMarkdownBatchArtifacts:
    """Verify AST artifact creation rules for single and batch extraction."""

    def test_single_markdown_writes_ast_and_metadata(self, tmp_path, monkeypatch):
        source = _write_markdown(tmp_path / "one.md", "# One\n\nBody.\n")
        output_text, output_meta, output_ast = _patch_outputs(
            monkeypatch, tmp_path / "output"
        )
        monkeypatch.setattr(
            "sys.argv", ["extract.py", str(source), "--no-install-missing"]
        )

        main()

        metadata = json.loads(output_meta.read_text(encoding="utf-8"))
        artifact = json.loads(output_ast.read_text(encoding="utf-8"))
        assert output_text.exists()
        assert metadata["output_ast"] == str(output_ast)
        assert metadata["markdown_source_count"] == 1
        assert metadata["markdown_ast_source_count"] == 1
        assert metadata["chapters_detected"] == 1
        assert metadata["sources"][0]["ast_available"] is True
        assert artifact["schema_version"] == "1.0"
        assert artifact["line_map_convention"] == "one-based inclusive"
        assert [source["filename"] for source in artifact["sources"]] == ["one.md"]

    def test_multiple_markdown_sources_share_one_ast_artifact(
        self, tmp_path, monkeypatch
    ):
        first = _write_markdown(tmp_path / "first.md", "# First\n")
        second = _write_markdown(tmp_path / "second.markdown", "# Second\n")
        _, output_meta, output_ast = _patch_outputs(monkeypatch, tmp_path / "output")
        monkeypatch.setattr("sys.argv", ["extract.py", str(first), str(second)])

        main()

        metadata = json.loads(output_meta.read_text(encoding="utf-8"))
        artifact = json.loads(output_ast.read_text(encoding="utf-8"))
        assert metadata["markdown_source_count"] == 2
        assert metadata["markdown_ast_source_count"] == 2
        assert [source["filename"] for source in artifact["sources"]] == [
            "first.md",
            "second.markdown",
        ]

    def test_mixed_batch_puts_only_markdown_in_ast_artifact(
        self, tmp_path, monkeypatch
    ):
        markdown = _write_markdown(tmp_path / "chapter.md", "# Chapter\n")
        plain = tmp_path / "notes.txt"
        plain.write_text("Plain notes.", encoding="utf-8")
        _, output_meta, output_ast = _patch_outputs(monkeypatch, tmp_path / "output")
        monkeypatch.setattr("sys.argv", ["extract.py", str(markdown), str(plain)])

        main()

        metadata = json.loads(output_meta.read_text(encoding="utf-8"))
        artifact = json.loads(output_ast.read_text(encoding="utf-8"))
        assert metadata["total_sources"] == 2
        assert metadata["markdown_source_count"] == 1
        assert [source["filename"] for source in artifact["sources"]] == ["chapter.md"]

    def test_non_markdown_batch_removes_stale_ast_artifact(self, tmp_path, monkeypatch):
        plain = tmp_path / "notes.txt"
        plain.write_text("Plain notes.", encoding="utf-8")
        _, output_meta, output_ast = _patch_outputs(monkeypatch, tmp_path / "output")
        output_ast.parent.mkdir(parents=True)
        output_ast.write_text("stale", encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["extract.py", str(plain)])

        main()

        metadata = json.loads(output_meta.read_text(encoding="utf-8"))
        assert metadata["output_ast"] is None
        assert metadata["markdown_source_count"] == 0
        assert metadata["markdown_ast_source_count"] == 0
        assert not output_ast.exists()
