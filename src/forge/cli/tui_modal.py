"""Interactive TUI Modal Dropdown Picker for Models and Providers.

Renders an interactive, keyboard-navigable popup modal matching modern CLI agent
design (OpenCode/Claude Code) with live search, section headers, peach/orange highlight bar,
and hotkeys (Esc to exit, Ctrl+A for provider connection).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style


@dataclass
class TuiItem:
    """An individual selectable item within the modal popup."""

    id: str
    title: str
    subtitle: str = ""
    section: str = ""
    badge: str = ""
    is_checked: bool = False
    is_dot: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


CTRL_A_TOKEN = "__CONNECT_PROVIDER__"

MODAL_STYLE = Style.from_dict({
    "backdrop": "bg:#000000",
    "card": "bg:#181818 fg:#ffffff",
    "card_bg": "bg:#181818",
    "title": "bg:#181818 fg:#ffffff bold",
    "esc": "bg:#181818 fg:#737373",
    "search_prompt": "bg:#181818 fg:#737373",
    "search_text": "bg:#181818 fg:#ffffff bold",
    "cursor": "bg:#181818 fg:#f5a770 bold",
    "section": "bg:#181818 fg:#c084fc bold",
    "item_normal": "bg:#181818 fg:#e5e5e5",
    "item_dim": "bg:#181818 fg:#737373",
    "item_badge": "bg:#181818 fg:#a3e635 bold",
    "item_check": "bg:#181818 fg:#22c55e bold",
    "item_dot": "bg:#181818 fg:#ffffff",
    "item_selected": "bg:#f5a770 fg:#141414 bold",
    "item_selected_dim": "bg:#f5a770 fg:#262626",
    "item_selected_badge": "bg:#f5a770 fg:#141414 bold",
    "footer_label": "bg:#181818 fg:#e5e5e5",
    "footer_key": "bg:#181818 fg:#737373",
    "no_matches": "bg:#181818 fg:#737373 italic",
})


def run_tui_modal(
    title: str,
    items: list[TuiItem],
    search_placeholder: str = "Search",
    footer_left: str = "Connect provider",
    footer_left_key: str = "ctrl+a",
    footer_right: str = "Favorite",
    footer_right_key: str = "ctrl+f",
    allow_ctrl_a: bool = True,
    allow_custom: bool = True,
    card_width: int = 62,
    visible_rows: int = 8,
    input_override: Input | None = None,
    output_override: Output | None = None,
) -> TuiItem | str | None:
    """Run an interactive TUI modal dialog and return the chosen item or special token.

    Returns:
        - TuiItem if an item was chosen by pressing Enter
        - CTRL_A_TOKEN if user pressed Ctrl+A
        - None if user cancelled with Escape or Ctrl+C
    """
    if not items and not allow_custom:
        return None

    # In headless / non-interactive environment without explicit input override, return None (cancelled)
    if input_override is None:
        try:
            if not sys.stdin.isatty():
                return None
        except Exception:
            return None

    search_query = ""
    selected_index = 0
    scroll_offset = 0

    def get_filtered_items() -> list[TuiItem]:
        q = search_query.strip().lower()
        if not q:
            return list(items)
        filtered = [
            it
            for it in items
            if q in it.title.lower()
            or q in it.subtitle.lower()
            or q in it.id.lower()
            or q in it.section.lower()
        ]
        if not filtered and allow_custom:
            return [
                TuiItem(
                    id=search_query.strip(),
                    title=f"Use \"{search_query.strip()}\"",
                    subtitle="Custom ID",
                    section="Custom",
                )
            ]
        return filtered

    def render_content() -> StyleAndTextTuples:
        nonlocal selected_index, scroll_offset
        filtered = get_filtered_items()
        n_items = len(filtered)
        if n_items == 0:
            selected_index = 0
        else:
            selected_index = max(0, min(selected_index, n_items - 1))

        # Adjust scroll offset so selected_index is within visible_rows
        if selected_index < scroll_offset:
            scroll_offset = selected_index
        elif selected_index >= scroll_offset + visible_rows:
            scroll_offset = selected_index - visible_rows + 1

        scroll_offset = max(0, scroll_offset)
        viewport = filtered[scroll_offset : scroll_offset + visible_rows]

        lines: StyleAndTextTuples = []

        # 1. Header line: Title on left, "esc" on right
        title_text = f"  {title}"
        esc_text = "esc  "
        gap = max(1, card_width - len(title_text) - len(esc_text))
        lines.append(("class:title", title_text))
        lines.append(("class:card_bg", " " * gap))
        lines.append(("class:esc", esc_text + "\n"))

        # 2. Search line
        if not search_query:
            prompt_text = f"  {search_placeholder}"
            lines.append(("class:search_prompt", prompt_text))
            lines.append(("class:card_bg", " " * max(0, card_width - len(prompt_text)) + "\n"))
        else:
            q_text = f"  {search_query}"
            lines.append(("class:search_text", q_text))
            lines.append(("class:cursor", "▏"))
            lines.append(("class:card_bg", " " * max(0, card_width - len(q_text) - 1) + "\n"))

        # Blank line
        lines.append(("class:card_bg", " " * card_width + "\n"))

        # 3. Items list with section headers
        last_section = None
        # Look back one item before viewport to know if current section was already started
        if scroll_offset > 0 and scroll_offset - 1 < len(filtered):
            last_section = filtered[scroll_offset - 1].section

        displayed_rows = 0
        for idx_in_viewport, item in enumerate(viewport):
            item_global_idx = scroll_offset + idx_in_viewport
            is_active = item_global_idx == selected_index

            # Render section header if section changed and section is not empty
            if item.section and item.section != last_section:
                sec_text = f"  {item.section}"
                lines.append(("class:section", sec_text))
                lines.append(("class:card_bg", " " * max(0, card_width - len(sec_text)) + "\n"))
                last_section = item.section
                displayed_rows += 1

            # Determine bullet/prefix
            if is_active:
                if item.is_dot:
                    prefix = "● "
                elif item.is_checked:
                    prefix = "✓ "
                else:
                    prefix = "  "
            else:
                if item.is_dot:
                    prefix = "● "
                elif item.is_checked:
                    prefix = "✓ "
                else:
                    prefix = "  "

            # Build line content
            badge_str = f"{item.badge}  " if item.badge else "  "
            badge_len = len(badge_str)

            # Left content: prefix + title + subtitle
            left_prefix = f"  {prefix}"
            title_part = item.title
            sub_part = f" {item.subtitle}" if item.subtitle else ""

            # Check fit
            avail_for_left = card_width - badge_len
            combined_left = f"{left_prefix}{title_part}{sub_part}"
            if len(combined_left) > avail_for_left:
                # Truncate subtitle or title
                overflow = len(combined_left) - avail_for_left + 1
                if len(sub_part) > overflow + 3:
                    sub_part = sub_part[: -overflow - 1] + "…"
                else:
                    title_part = title_part[: max(4, len(title_part) - overflow - 1)] + "…"
                combined_left = f"{left_prefix}{title_part}{sub_part}"

            fill_spaces = " " * max(0, avail_for_left - len(combined_left))

            if is_active:
                # Peach/orange highlight bar across entire row
                lines.append(("class:item_selected", left_prefix + title_part))
                if sub_part:
                    lines.append(("class:item_selected_dim", sub_part))
                lines.append(("class:item_selected", fill_spaces))
                if item.badge:
                    lines.append(("class:item_selected_badge", badge_str))
                else:
                    lines.append(("class:item_selected", "  "))
                lines.append(("class:card_bg", "\n"))
            else:
                # Normal row
                if item.is_checked:
                    lines.append(("class:card_bg", "  "))
                    lines.append(("class:item_check", prefix))
                elif item.is_dot:
                    lines.append(("class:card_bg", "  "))
                    lines.append(("class:item_dot", prefix))
                else:
                    lines.append(("class:card_bg", left_prefix))

                lines.append(("class:item_normal", title_part))
                if sub_part:
                    lines.append(("class:item_dim", sub_part))
                lines.append(("class:card_bg", fill_spaces))
                if item.badge:
                    lines.append(("class:item_badge", badge_str))
                else:
                    lines.append(("class:card_bg", "  "))
                lines.append(("class:card_bg", "\n"))

            displayed_rows += 1

        if n_items == 0:
            no_text = "  No matching items found."
            lines.append(("class:no_matches", no_text))
            lines.append(("class:card_bg", " " * max(0, card_width - len(no_text)) + "\n"))
            displayed_rows += 1

        # Fill remaining rows up to visible_rows to maintain stable card height
        needed_padding = max(0, visible_rows - displayed_rows)
        for _ in range(needed_padding):
            lines.append(("class:card_bg", " " * card_width + "\n"))

        # Blank line
        lines.append(("class:card_bg", " " * card_width + "\n"))

        # 4. Footer line
        f_left = f"  {footer_left} "
        f_left_k = f"{footer_left_key}   "
        f_right = f"{footer_right} "
        f_right_k = f"{footer_right_key}  "
        used = len(f_left) + len(f_left_k) + len(f_right) + len(f_right_k)
        footer_gap = " " * max(1, card_width - used)

        lines.append(("class:footer_label", f_left))
        lines.append(("class:footer_key", f_left_k))
        lines.append(("class:footer_label", f_right))
        lines.append(("class:footer_key", f_right_k))
        lines.append(("class:card_bg", footer_gap + "\n"))

        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _up(event: KeyPressEvent) -> None:
        nonlocal selected_index
        filtered = get_filtered_items()
        if filtered:
            selected_index = (selected_index - 1) % len(filtered)

    @kb.add("down")
    def _down(event: KeyPressEvent) -> None:
        nonlocal selected_index
        filtered = get_filtered_items()
        if filtered:
            selected_index = (selected_index + 1) % len(filtered)

    @kb.add("pageup")
    def _pageup(event: KeyPressEvent) -> None:
        nonlocal selected_index
        filtered = get_filtered_items()
        if filtered:
            selected_index = max(0, selected_index - 5)

    @kb.add("pagedown")
    def _pagedown(event: KeyPressEvent) -> None:
        nonlocal selected_index
        filtered = get_filtered_items()
        if filtered:
            selected_index = min(len(filtered) - 1, selected_index + 5)

    @kb.add("enter")
    def _enter(event: KeyPressEvent) -> None:
        filtered = get_filtered_items()
        if filtered and 0 <= selected_index < len(filtered):
            event.app.exit(result=filtered[selected_index])
        elif allow_custom and search_query.strip():
            event.app.exit(
                result=TuiItem(
                    id=search_query.strip(),
                    title=search_query.strip(),
                    subtitle="Custom",
                )
            )
        else:
            event.app.exit(result=None)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    if allow_ctrl_a:
        @kb.add("c-a")
        def _ctrl_a(event: KeyPressEvent) -> None:
            event.app.exit(result=CTRL_A_TOKEN)

    @kb.add("backspace")
    def _backspace(event: KeyPressEvent) -> None:
        nonlocal search_query, selected_index
        if search_query:
            search_query = search_query[:-1]
            selected_index = 0

    @kb.add("<any>")
    def _any(event: KeyPressEvent) -> None:
        nonlocal search_query, selected_index
        char = event.data
        if char and char.isprintable():
            search_query += char
            selected_index = 0

    # Layout construction: modal centered in screen
    card_window = Window(
        content=FormattedTextControl(render_content),
        width=Dimension.exact(card_width),
        height=Dimension.exact(visible_rows + 7),
        style="class:card",
    )

    root_container = HSplit([
        Window(),  # Top spacer
        VSplit([
            Window(),  # Left spacer
            card_window,
            Window(),  # Right spacer
        ]),
        Window(),  # Bottom spacer
    ], style="class:backdrop")

    layout = Layout(root_container)

    try:
        app: Application[Any] = Application(
            layout=layout,
            key_bindings=kb,
            style=MODAL_STYLE,
            full_screen=True,
            input=input_override,
            output=output_override,
        )
        res = app.run()
        if isinstance(res, (TuiItem, str)) or res is None:
            return res
        return None
    except Exception:
        # Fallback for non-interactive / headless environments
        return None
