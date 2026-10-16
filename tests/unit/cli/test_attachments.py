"""Unit tests for multimodal attachment handling and clipboard extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
from rich.panel import Panel

from forge.cli.attachments import (
    build_turn_content,
    create_attachment_from_file,
    get_clipboard_files,
    get_clipboard_image,
    parse_user_input_attachments,
    take_screenshot,
)
from forge.models.types import ImagePart, TextPart


def _create_dummy_image(path: Path, width: int = 40, height: int = 30) -> Path:
    img = Image.new("RGB", (width, height), color="blue")
    img.save(path, format="PNG")
    return path


class TestAttachments:
    def test_create_attachment_from_image(self, tmp_path: Path) -> None:
        img_path = _create_dummy_image(tmp_path / "test_screen.png", 64, 48)
        att = create_attachment_from_file(img_path)
        assert att is not None
        assert att.is_image
        assert not att.is_code
        assert att.mime_type == "image/png"
        assert att.width == 64
        assert att.height == 48
        assert att.size_bytes > 0
        assert "64x48" in att.description

        # Data URI test
        data_uri = att.to_data_uri()
        assert data_uri.startswith("data:image/png;base64,")
        assert len(data_uri) > 30

        # Rich panel test
        panel = att.render_panel()
        assert isinstance(panel, Panel)

    def test_create_attachment_from_code(self, tmp_path: Path) -> None:
        code_path = tmp_path / "solution.py"
        code_path.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        att = create_attachment_from_file(code_path)
        assert att is not None
        assert att.is_code
        assert not att.is_image
        assert att.mime_type == "text/x-python" or "text" in att.mime_type

        # Text block test
        block = att.to_text_block()
        assert "def hello():" in block
        assert "solution.py" in block

        # Rich panel test
        panel = att.render_panel()
        assert isinstance(panel, Panel)

    def test_create_attachment_from_nonexistent_returns_none(self, tmp_path: Path) -> None:
        assert create_attachment_from_file(tmp_path / "does_not_exist.png") is None

    def test_parse_user_input_with_tags(self, tmp_path: Path) -> None:
        img_path = _create_dummy_image(tmp_path / "error.png")
        input_str = f"Explain this error [image: {img_path.as_posix()}] please"
        cleaned_text, atts = parse_user_input_attachments(input_str, base_dir=tmp_path)
        assert len(atts) == 1
        assert atts[0].path == img_path
        assert "Explain this error" in cleaned_text
        assert "please" in cleaned_text
        assert "[image:" not in cleaned_text

    def test_parse_user_input_with_quoted_path(self, tmp_path: Path) -> None:
        img_path = _create_dummy_image(tmp_path / "ui_bug.png")
        input_str = f'Look at "{img_path.as_posix()}" and find the CSS error'
        cleaned_text, atts = parse_user_input_attachments(input_str, base_dir=tmp_path)
        assert len(atts) == 1
        assert atts[0].path == img_path
        assert "attached image (ui_bug.png)" in cleaned_text
        assert "CSS error" in cleaned_text

    def test_build_turn_content_with_image(self, tmp_path: Path) -> None:
        img_path = _create_dummy_image(tmp_path / "screenshot.png")
        att = create_attachment_from_file(img_path)
        assert att is not None

        content, att_list = build_turn_content("Diagnose this crash", [att])
        assert isinstance(content, list)
        assert len(content) == 2
        assert isinstance(content[0], TextPart)
        assert "Diagnose this crash" in content[0].text
        assert isinstance(content[1], ImagePart)
        assert content[1].url.startswith("data:image/png;base64,")
        assert len(att_list) == 1

    def test_build_turn_content_with_code_only(self, tmp_path: Path) -> None:
        code_path = tmp_path / "script.py"
        code_path.write_text("print('test')", encoding="utf-8")
        att = create_attachment_from_file(code_path)
        assert att is not None

        content, att_list = build_turn_content("Review this script", [att])
        assert isinstance(content, str)
        assert "Review this script" in content
        assert "print('test')" in content
        assert len(att_list) == 1

    def test_build_turn_content_empty_attachments(self) -> None:
        content, att_list = build_turn_content("Plain question", [])
        assert content == "Plain question"
        assert att_list == []

    def test_get_clipboard_image_from_pil(self, tmp_path: Path) -> None:
        mock_pil_img = Image.new("RGB", (50, 50), color="red")
        with patch("PIL.ImageGrab.grabclipboard", return_value=mock_pil_img):
            att = get_clipboard_image(save_dir=tmp_path)
            assert att is not None
            assert att.is_image
            assert att.width == 50
            assert att.height == 50
            assert att.path.is_file()

    def test_get_clipboard_image_from_copied_file_list(self, tmp_path: Path) -> None:
        img_path = _create_dummy_image(tmp_path / "copied.png")
        with patch("PIL.ImageGrab.grabclipboard", return_value=[str(img_path)]):
            att = get_clipboard_image(save_dir=tmp_path / "saved")
            assert att is not None
            assert att.is_image
            assert "copied.png" in att.path.name

    def test_get_clipboard_image_none(self, tmp_path: Path) -> None:
        with (
            patch("PIL.ImageGrab.grabclipboard", return_value=None),
            patch("subprocess.run") as mock_subproc,
        ):
            mock_subproc.return_value = MagicMock(stdout="", returncode=0)
            att = get_clipboard_image(save_dir=tmp_path)
            assert att is None

    def test_get_clipboard_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "file1.py"
        f1.write_text("x = 1")
        f2 = tmp_path / "file2.json"
        f2.write_text("{}")
        with patch("PIL.ImageGrab.grabclipboard", return_value=[str(f1), str(f2)]):
            files = get_clipboard_files()
            assert len(files) == 2
            assert any(f.path == f1 for f in files)
            assert any(f.path == f2 for f in files)

    def test_take_screenshot(self, tmp_path: Path) -> None:
        mock_pil_img = Image.new("RGB", (100, 100), color="green")
        with patch("PIL.ImageGrab.grab", return_value=mock_pil_img):
            att = take_screenshot(save_dir=tmp_path)
            assert att is not None
            assert att.is_image
            assert att.width == 100
            assert att.height == 100
            assert att.path.is_file()
