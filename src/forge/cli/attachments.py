"""Multimodal and file attachments handling for Forge CLI.

Enables developers to paste screenshots, drag-and-drop images/files, or attach code
directly into the Forge interactive chat. Handles clipboard extraction across Windows,
macOS, and Linux, base64 data-URI encoding, PIL resizing optimization, and Rich preview cards.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.panel import Panel

from forge.models.types import ContentPart, ImagePart, TextPart

IMAGE_EXTENSIONS = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".svg",
    ".tiff",
    ".ico",
})

CODE_EXTENSIONS = frozenset({
    ".py",
    ".ts",
    ".js",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".sh",
    ".bash",
    ".sql",
    ".rs",
    ".go",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".java",
    ".kt",
    ".swift",
    ".log",
    ".xml",
    ".ini",
    ".env",
})

DOCUMENT_EXTENSIONS = frozenset({
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".csv",
})


@dataclass
class Attachment:
    """An attached file or screenshot image."""

    path: Path
    kind: Literal["image", "code", "document", "other"]
    mime_type: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    description: str = ""

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    @property
    def is_code(self) -> bool:
        return self.kind == "code"

    @property
    def is_document(self) -> bool:
        return self.kind == "document"

    def to_data_uri(
        self,
        max_dimension: int = 2048,
        max_size_bytes: int = 4 * 1024 * 1024,
    ) -> str:
        """Encode the image file as a base64 data URI, optimizing if excessively large."""
        if not self.path.is_file():
            raise FileNotFoundError(f"Attachment file not found: {self.path}")

        raw_bytes = self.path.read_bytes()
        mime = self.mime_type or "image/png"

        try:
            from PIL import Image

            with Image.open(io.BytesIO(raw_bytes)) as img:
                w, h = img.size
                if w > max_dimension or h > max_dimension or len(raw_bytes) > max_size_bytes:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    if img.mode in ("RGBA", "LA") or (
                        img.mode == "P" and "transparency" in img.info
                    ):
                        img.save(buf, format="PNG", optimize=True)
                        mime = "image/png"
                    else:
                        rgb = img.convert("RGB")
                        rgb.save(buf, format="JPEG", quality=85, optimize=True)
                        mime = "image/jpeg"
                    raw_bytes = buf.getvalue()
        except Exception:
            pass

        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def to_text_block(self) -> str:
        """Read and format text/code content for injection into model prompts."""
        if not self.path.is_file():
            return f"--- Attached File: {self.path.name} (File not found) ---\n"

        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
            # If exceedingly large (>250KB), truncate to preserve context window
            if len(text) > 250_000:
                text = text[:250_000] + "\n\n... [Content truncated at 250KB to preserve context window] ..."
            ext = self.path.suffix.lstrip(".").lower() or "txt"
            return (
                f"--- Attached File: `{self.path.name}` ({self.kind}) ---\n"
                f"```{ext}\n"
                f"{text}\n"
                f"```\n"
                f"--- End of `{self.path.name}` ---\n"
            )
        except Exception as exc:
            return f"--- Attached File: {self.path.name} (Error reading file: {exc}) ---\n"

    def render_panel(self) -> Panel:
        """Generate a sleek, formatted Rich Panel for terminal display."""
        size_kb = round(self.size_bytes / 1024, 1)
        if self.is_image:
            dim_str = f"{self.width} × {self.height}" if self.width and self.height else "detected"
            body = (
                f"[bold cyan]📷 Screenshot / Image Attached:[/bold cyan] [bold]{self.path.name}[/bold]\n"
                f"[dim]Dimensions:[/dim] {dim_str} px  •  [dim]Size:[/dim] {size_kb} KB  •  [dim]Type:[/dim] {self.mime_type}\n"
                f"[dim]Path:[/dim] {self.path.as_posix()}"
            )
            return Panel(body, border_style="cyan", padding=(0, 1))

        if self.is_code:
            body = (
                f"[bold yellow]📄 Code File Attached:[/bold yellow] [bold]{self.path.name}[/bold]\n"
                f"[dim]Size:[/dim] {size_kb} KB  •  [dim]Path:[/dim] {self.path.as_posix()}"
            )
            return Panel(body, border_style="yellow", padding=(0, 1))

        body = (
            f"[bold green]📑 Document Attached:[/bold green] [bold]{self.path.name}[/bold]\n"
            f"[dim]Size:[/dim] {size_kb} KB  •  [dim]Path:[/dim] {self.path.as_posix()}"
        )
        return Panel(body, border_style="green", padding=(0, 1))


def create_attachment_from_file(file_path: Path | str) -> Attachment | None:
    """Create an Attachment metadata object from an existing filesystem path."""
    p = Path(file_path).resolve()
    if not p.is_file():
        return None

    ext = p.suffix.lower()
    size = p.stat().st_size
    mime, _ = mimetypes.guess_type(p.name)

    if ext in IMAGE_EXTENSIONS:
        width = None
        height = None
        try:
            from PIL import Image

            with Image.open(p) as img:
                width, height = img.size
        except Exception:
            pass

        return Attachment(
            path=p,
            kind="image",
            mime_type=mime or "image/png",
            size_bytes=size,
            width=width,
            height=height,
            description=f"Image {p.name} ({width}x{height})",
        )

    if ext in CODE_EXTENSIONS:
        return Attachment(
            path=p,
            kind="code",
            mime_type=mime or "text/plain",
            size_bytes=size,
            description=f"Code file {p.name}",
        )

    if ext in DOCUMENT_EXTENSIONS:
        return Attachment(
            path=p,
            kind="document",
            mime_type=mime or "application/octet-stream",
            size_bytes=size,
            description=f"Document {p.name}",
        )

    return Attachment(
        path=p,
        kind="other",
        mime_type=mime or "application/octet-stream",
        size_bytes=size,
        description=f"File {p.name}",
    )


def _get_default_media_dir(custom_dir: Path | None = None) -> Path:
    """Ensure and return the directory used to persist pasted screenshots."""
    if custom_dir is not None:
        target = custom_dir
    else:
        # Default to .forge/media in active workspace, fallback to ~/.forge/media
        cwd_forge = Path.cwd() / ".forge" / "media"
        target = cwd_forge if cwd_forge.parent.is_dir() else Path.home() / ".forge" / "media"

    target.mkdir(parents=True, exist_ok=True)
    return target


def get_clipboard_image(save_dir: Path | None = None) -> Attachment | None:
    """Extract an image currently present in the system clipboard (cross-platform)."""
    dest_dir = _get_default_media_dir(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Primary Method: Pillow ImageGrab
    try:
        from PIL import Image, ImageGrab

        clip = ImageGrab.grabclipboard()
        if clip is not None:
            # Clipboard contains raw image data (screenshot, copied bitmap)
            if isinstance(clip, Image.Image):
                out_path = dest_dir / f"screenshot_{timestamp}.png"
                clip.save(out_path, format="PNG")
                return create_attachment_from_file(out_path)

            # Clipboard contains a list of file paths (e.g. copied from File Explorer / Finder)
            if isinstance(clip, list):
                for item in clip:
                    p = Path(item)
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                        copied_target = dest_dir / f"screenshot_{timestamp}_{p.name}"
                        shutil.copy2(p, copied_target)
                        return create_attachment_from_file(copied_target)
    except Exception:
        pass

    # 2. Windows Fallback: PowerShell Clipboard Forms extraction
    if sys.platform == "win32":
        try:
            target_path = dest_dir / f"screenshot_{timestamp}.png"
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "if ([System.Windows.Forms.Clipboard]::ContainsImage()) { "
                "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
                f"$img.Save('{target_path.as_posix()}', [System.Drawing.Imaging.ImageFormat]::Png); "
                "Write-Output 'OK' "
                "} else { "
                "$files = [System.Windows.Forms.Clipboard]::GetFileDropList(); "
                "if ($files -and $files.Count -gt 0) { Write-Output $files[0] } "
                "}"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            out = res.stdout.strip()
            if "OK" in out and target_path.is_file():
                return create_attachment_from_file(target_path)
            if out and Path(out).is_file() and Path(out).suffix.lower() in IMAGE_EXTENSIONS:
                copied_target = dest_dir / f"screenshot_{timestamp}_{Path(out).name}"
                shutil.copy2(Path(out), copied_target)
                return create_attachment_from_file(copied_target)
        except Exception:
            pass

    # 3. macOS Fallback: osascript / pngpaste
    if sys.platform == "darwin":
        try:
            target_path = dest_dir / f"screenshot_{timestamp}.png"
            cmd = [
                "osascript",
                "-e",
                f'set theFile to open for access POSIX file "{target_path.as_posix()}" with write permission',
                "-e",
                "write (get the clipboard as «class PNGf») to theFile",
                "-e",
                "close access theFile",
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=5)
            if res.returncode == 0 and target_path.is_file() and target_path.stat().st_size > 0:
                return create_attachment_from_file(target_path)
        except Exception:
            pass

    # 4. Linux Fallback: wl-paste (Wayland) / xclip (X11)
    if sys.platform.startswith("linux"):
        target_path = dest_dir / f"screenshot_{timestamp}.png"
        for binary, args in [
            ("wl-paste", ["--type", "image/png"]),
            ("xclip", ["-selection", "clipboard", "-t", "image/png", "-out"]),
        ]:
            if shutil.which(binary):
                try:
                    res = subprocess.run([binary, *args], capture_output=True, timeout=5)
                    if res.returncode == 0 and res.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
                        target_path.write_bytes(res.stdout)
                        return create_attachment_from_file(target_path)
                except Exception:
                    pass

    return None


def get_clipboard_files() -> list[Attachment]:
    """Extract file paths copied to the system clipboard (e.g. from File Explorer/Finder)."""
    attachments: list[Attachment] = []

    # 1. Pillow ImageGrab
    try:
        from PIL import ImageGrab

        clip = ImageGrab.grabclipboard()
        if isinstance(clip, list):
            for item in clip:
                att = create_attachment_from_file(item)
                if att is not None:
                    attachments.append(att)
            if attachments:
                return attachments
    except Exception:
        pass

    # 2. Windows PowerShell GetFileDropList
    if sys.platform == "win32":
        try:
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$files = [System.Windows.Forms.Clipboard]::GetFileDropList(); "
                "foreach ($f in $files) { Write-Output $f }"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in res.stdout.splitlines():
                line = line.strip()
                if line:
                    att = create_attachment_from_file(line)
                    if att is not None:
                        attachments.append(att)
        except Exception:
            pass

    return attachments


def take_screenshot(save_dir: Path | None = None) -> Attachment | None:
    """Capture a live full-screen screenshot of the desktop."""
    dest_dir = _get_default_media_dir(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = dest_dir / f"screenshot_full_{timestamp}.png"

    try:
        from PIL import ImageGrab

        img = ImageGrab.grab()
        img.save(out_path, format="PNG")
        return create_attachment_from_file(out_path)
    except Exception:
        pass

    # Fallback to checking active clipboard
    return get_clipboard_image(save_dir)


def parse_user_input_attachments(
    user_input: str,
    base_dir: Path | None = None,
) -> tuple[str, list[Attachment]]:
    """Scan user input text for attachment tags or file paths and return cleaned text + attachments.

    Recognizes:
    - [image: /path/to/screenshot.png] or [image: ./relative/image.png]
    - [file: /path/to/source.py]
    - Standalone or quoted file paths like "C:\\Users\\Pictures\\error.png"
    - Dropped terminal paths ending in known image/code/document extensions.
    """
    root = (base_dir or Path.cwd()).resolve()
    attachments: list[Attachment] = []
    seen_paths: set[str] = set()

    def _try_add(candidate: str | Path) -> Attachment | None:
        p_cand = Path(candidate)
        resolved = (p_cand if p_cand.is_absolute() else root / p_cand).resolve()
        if resolved.is_file() and str(resolved) not in seen_paths:
            att = create_attachment_from_file(resolved)
            if att:
                seen_paths.add(str(resolved))
                attachments.append(att)
                return att
        return None

    cleaned_text = user_input

    # 1. Match [image: ...] or [file: ...] tags
    tag_pattern = re.compile(r"\[(image|file):\s*([^\]]+)\]", re.IGNORECASE)
    for match in tag_pattern.finditer(user_input):
        raw_path = match.group(2).strip().strip("'\"")
        _try_add(raw_path)

    cleaned_text = tag_pattern.sub("", cleaned_text).strip()

    # 2. Match quoted file paths: "C:\path\to\file.png" or 'relative/file.py'
    quoted_pattern = re.compile(r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']')
    for match in quoted_pattern.finditer(cleaned_text):
        raw_path = match.group(1).strip()
        att = _try_add(raw_path)
        if att and att.is_image:
            # Remove the full image path from the query if it was dragged or quoted
            cleaned_text = cleaned_text.replace(match.group(0), f"attached image ({att.path.name})")

    # 3. Match standalone unquoted file path if the whole line or first token is a file
    tokens = cleaned_text.split()
    for token in tokens:
        clean_tok = token.strip(" \t\n,;:")
        ext = Path(clean_tok).suffix.lower()
        if ext in (IMAGE_EXTENSIONS | CODE_EXTENSIONS | DOCUMENT_EXTENSIONS):
            att = _try_add(clean_tok)
            if att and att.is_image:
                cleaned_text = cleaned_text.replace(token, f"attached image ({att.path.name})")

    # Normalize whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text, attachments


def build_turn_content(
    user_text: str,
    attachments: list[Attachment],
) -> tuple[str | list[ContentPart], list[Attachment]]:
    """Assemble multimodal ContentPart list or formatted text for the active agent turn."""
    if not attachments:
        return user_text, []

    has_images = any(a.is_image for a in attachments)

    if has_images:
        parts: list[ContentPart] = []

        # Construct textual prompt
        text_blocks: list[str] = []
        if user_text:
            text_blocks.append(user_text)
        else:
            text_blocks.append("Please analyze the attached image and diagnose any errors or details.")

        for a in attachments:
            if not a.is_image:
                text_blocks.append(a.to_text_block())

        prompt_text = "\n\n".join(text_blocks).strip()
        parts.append(TextPart(text=prompt_text))

        # Append image parts
        for a in attachments:
            if a.is_image:
                try:
                    data_uri = a.to_data_uri()
                    parts.append(ImagePart(url=data_uri))
                except Exception as exc:
                    prompt_text += f"\n\n[Warning: Could not encode image {a.path.name}: {exc}]"
                    parts[0] = TextPart(text=prompt_text)

        return parts, attachments

    # Only code or document files: embed directly into markdown prompt text
    text_blocks = []
    if user_text:
        text_blocks.append(user_text)
    for a in attachments:
        text_blocks.append(a.to_text_block())

    return "\n\n".join(text_blocks).strip(), attachments
