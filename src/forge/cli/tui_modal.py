"""Interactive TUI Modal Dropdown Picker for Models and Providers.

Renders an elevated, keyboard-navigable card dialog styled in Forge Cyan and Green
with live fuzzy search, match counters, framed borders, section headers, active
highlight bar, scroll overflow hints, and hotkey action chips.
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


@dataclass
class _DisplayEntry:
    """Internal flattened entry for rendering rows (section headers or selectable items)."""

    is_section: bool
    section_name: str = ""
    item: TuiItem | None = None
    item_index: int = -1  # Index in the filtered list of selectable items


CTRL_A_TOKEN = "__CONNECT_PROVIDER__"


def _can_encode(text: str) -> bool:
    """Check if the current terminal stdout encoding can safely encode text."""
    try:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        text.encode(enc)
        return True
    except Exception:
        return False


# Determine safe character set based on terminal stdout capability
if _can_encode("╭─╮│╰─╯├┤▲▼●❯✓✦◈◆⌕"):
    BORDER_TL = "╭"
    BORDER_TR = "╮"
    BORDER_BL = "╰"
    BORDER_BR = "╯"
    BORDER_H = "─"
    BORDER_V = "│"
    BORDER_LT = "├"
    BORDER_RT = "┤"
    ARROW_UP = "▲"
    ARROW_DOWN = "▼"
    DOT_ACTIVE = "●"
    POINTER = "❯"
    SEARCH_ICON = "⌕"
    BRAND_ICON = "✦"
    SECTION_ICON = "◆"
    CHECK_ICON = "✓"
else:
    BORDER_TL = "+"
    BORDER_TR = "+"
    BORDER_BL = "+"
    BORDER_BR = "+"
    BORDER_H = "-"
    BORDER_V = "|"
    BORDER_LT = "+"
    BORDER_RT = "+"
    ARROW_UP = "^"
    ARROW_DOWN = "v"
    DOT_ACTIVE = "*"
    POINTER = ">"
    SEARCH_ICON = "/"
    BRAND_ICON = "*"
    SECTION_ICON = ">"
    CHECK_ICON = "v"


MODAL_STYLE = Style.from_dict({
    # Backdrop behind the floating modal card
    "backdrop": "bg:#070b12",
    # Modal card container
    "card": "bg:#0d1117 fg:#f1f5f9",
    "card_bg": "bg:#0d1117",
    # Frame borders
    "border": "bg:#0d1117 fg:#1e293b",
    "border_cyan": "bg:#0d1117 fg:#0891b2",
    # Top brand & title
    "brand_bolt": "bg:#0d1117 fg:#00e5ff bold",
    "brand_name": "bg:#0d1117 fg:#38bdf8 bold",
    "title_sep": "bg:#0d1117 fg:#475569",
    "title_text": "bg:#0d1117 fg:#ffffff bold",
    "esc_pill": "bg:#0d1117 fg:#64748b",
    # Search row
    "search_icon": "bg:#0d1117 fg:#00e5ff bold",
    "search_prompt": "bg:#0d1117 fg:#64748b italic",
    "search_text": "bg:#0d1117 fg:#ffffff bold",
    "cursor": "bg:#0d1117 fg:#00e5ff bold",
    "counter_pill": "bg:#0d1117 fg:#64748b",
    "counter_highlight": "bg:#0d1117 fg:#00e5ff bold",
    "counter_none": "bg:#0d1117 fg:#f59e0b italic",
    # Section headers
    "section_icon": "bg:#0d1117 fg:#00e5ff",
    "section_title": "bg:#0d1117 fg:#38bdf8 bold",
    # Scroll hints
    "scroll_hint": "bg:#0d1117 fg:#64748b",
    # Normal item row
    "item_normal": "bg:#0d1117 fg:#f1f5f9",
    "item_dim": "bg:#0d1117 fg:#64748b",
    "item_badge": "bg:#0d1117 fg:#06b6d4",
    "item_badge_green": "bg:#0d1117 fg:#10b981 bold",
    "item_check": "bg:#0d1117 fg:#10b981 bold",
    "item_dot": "bg:#0d1117 fg:#10b981 bold",
    # Selected/Active item row (glowing Forge cyan highlight bar)
    "pointer_active": "bg:#06b6d4 fg:#031d28 bold",
    "item_selected": "bg:#06b6d4 fg:#031d28 bold",
    "item_selected_dim": "bg:#06b6d4 fg:#164e63",
    "item_selected_badge": "bg:#06b6d4 fg:#031d28 bold",
    "item_selected_check": "bg:#06b6d4 fg:#031d28 bold",
    "item_selected_dot": "bg:#06b6d4 fg:#031d28 bold",
    # Footer toolbar
    "footer_key": "bg:#0d1117 fg:#00e5ff bold",
    "footer_label": "bg:#0d1117 fg:#94a3b8",
    "no_matches": "bg:#0d1117 fg:#94a3b8 italic",
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
    card_width: int = 70,
    visible_rows: int = 10,
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
        terms = q.split()
        filtered = [
            it
            for it in items
            if all(
                term in f"{it.title} {it.subtitle} {it.id} {it.section} {it.badge}".lower()
                for term in terms
            )
        ]
        if not filtered and allow_custom:
            return [
                TuiItem(
                    id=search_query.strip(),
                    title=f'Use "{search_query.strip()}"',
                    subtitle="Custom ID",
                    section="Custom Model",
                    badge="Custom",
                )
            ]
        return filtered

    def build_display_entries(filtered: list[TuiItem]) -> list[_DisplayEntry]:
        entries: list[_DisplayEntry] = []
        last_section: str | None = None
        for idx, it in enumerate(filtered):
            if it.section and it.section != last_section:
                entries.append(_DisplayEntry(is_section=True, section_name=it.section))
                last_section = it.section
            entries.append(_DisplayEntry(is_section=False, item=it, item_index=idx))
        return entries

    def render_content() -> StyleAndTextTuples:
        nonlocal selected_index, scroll_offset
        filtered = get_filtered_items()
        n_items = len(filtered)
        if n_items == 0:
            selected_index = 0
        else:
            selected_index = max(0, min(selected_index, n_items - 1))

        display_entries = build_display_entries(filtered)
        n_entries = len(display_entries)

        # Locate the display row for current selected_index
        selected_display_idx = 0
        for d_idx, entry in enumerate(display_entries):
            if not entry.is_section and entry.item_index == selected_index:
                selected_display_idx = d_idx
                break

        # Adjust scroll offset within display_entries
        if selected_display_idx < scroll_offset:
            scroll_offset = selected_display_idx
            # If the entry immediately preceding is its section header, show it too
            if scroll_offset > 0 and display_entries[scroll_offset - 1].is_section:
                scroll_offset -= 1
        elif selected_display_idx >= scroll_offset + visible_rows:
            scroll_offset = selected_display_idx - visible_rows + 1

        max_scroll = max(0, n_entries - visible_rows)
        scroll_offset = max(0, min(scroll_offset, max_scroll))

        viewport = display_entries[scroll_offset : scroll_offset + visible_rows]

        inner_w = card_width - 4  # Content width between left and right borders
        lines: StyleAndTextTuples = []

        # 1. Top border
        lines.append(("class:border", BORDER_TL + (BORDER_H * (card_width - 2)) + BORDER_TR + "\n"))

        # 2. Header bar: Brand bolt, Title, and Esc dismiss chip
        esc_chip = "[Esc: Cancel] "
        brand_icon_str = f" {BRAND_ICON} "
        brand_name_str = "FORGE "
        sep_str = "· "
        title_text = title

        used_left = len(brand_icon_str) + len(brand_name_str) + len(sep_str) + len(title_text)
        gap = inner_w - used_left - len(esc_chip)
        if gap < 1:
            overflow = 1 - gap
            title_text = title_text[: max(4, len(title_text) - overflow - 1)] + "…"
            used_left = len(brand_icon_str) + len(brand_name_str) + len(sep_str) + len(title_text)
            gap = max(1, inner_w - used_left - len(esc_chip))

        lines.append(("class:border", BORDER_V + " "))
        lines.append(("class:brand_bolt", brand_icon_str))
        lines.append(("class:brand_name", brand_name_str))
        lines.append(("class:title_sep", sep_str))
        lines.append(("class:title_text", title_text))
        lines.append(("class:card_bg", " " * gap))
        lines.append(("class:esc_pill", esc_chip))
        lines.append(("class:border", " " + BORDER_V + "\n"))

        # 3. Separator below header
        lines.append(("class:border", BORDER_LT + (BORDER_H * (card_width - 2)) + BORDER_RT + "\n"))

        # 4. Search input bar with live match counter
        search_icon_str = f" {SEARCH_ICON}  "
        if search_query:
            if n_items == 1:
                counter_str = "[ 1 match ] "
            elif n_items > 1:
                counter_str = f"[ {n_items} matches ] "
            else:
                counter_str = "[ 0 matches ] "
        else:
            counter_str = f"[ {len(items)} available ] "

        if not search_query:
            prompt_str = search_placeholder
            left_len = len(search_icon_str) + len(prompt_str)
            search_gap = max(1, inner_w - left_len - len(counter_str))
            lines.append(("class:border", BORDER_V + " "))
            lines.append(("class:search_icon", search_icon_str))
            lines.append(("class:search_prompt", prompt_str))
            lines.append(("class:card_bg", " " * search_gap))
            lines.append(("class:counter_pill", counter_str))
            lines.append(("class:border", " " + BORDER_V + "\n"))
        else:
            q_str = search_query
            left_len = len(search_icon_str) + len(q_str) + 1  # +1 for cursor
            search_gap = max(1, inner_w - left_len - len(counter_str))
            lines.append(("class:border", BORDER_V + " "))
            lines.append(("class:search_icon", search_icon_str))
            lines.append(("class:search_text", q_str))
            lines.append(("class:cursor", "▏"))
            lines.append(("class:card_bg", " " * search_gap))
            counter_style = "class:counter_highlight" if n_items > 0 else "class:counter_none"
            lines.append((counter_style, counter_str))
            lines.append(("class:border", " " + BORDER_V + "\n"))

        # 5. Separator above viewport (with Scroll Up indicator if scrolled)
        if scroll_offset > 0:
            hint = f" {ARROW_UP} {scroll_offset} more above "
            rem = (card_width - 2) - 2 - len(hint)
            lines.append(("class:border", BORDER_LT + (BORDER_H * 2)))
            lines.append(("class:scroll_hint", hint))
            lines.append(("class:border", (BORDER_H * max(0, rem)) + BORDER_RT + "\n"))
        else:
            lines.append(("class:border", BORDER_LT + (BORDER_H * (card_width - 2)) + BORDER_RT + "\n"))

        # 6. Items viewport rows
        displayed_rows = 0
        for entry in viewport:
            if entry.is_section:
                sec_icon_str = f" {SECTION_ICON} "
                sec_title = entry.section_name.upper()
                sec_fill = max(0, inner_w - len(sec_icon_str) - len(sec_title))
                lines.append(("class:border", BORDER_V + " "))
                lines.append(("class:section_icon", sec_icon_str))
                lines.append(("class:section_title", sec_title))
                lines.append(("class:card_bg", " " * sec_fill))
                lines.append(("class:border", " " + BORDER_V + "\n"))
            elif entry.item is not None:
                item = entry.item
                is_active = (entry.item_index == selected_index)

                pointer_str = f" {POINTER} " if is_active else "   "

                if item.is_dot:
                    dot_str = f"{DOT_ACTIVE} "
                elif item.is_checked:
                    dot_str = f"{CHECK_ICON} "
                else:
                    dot_str = "  "

                badge_str = f"[{item.badge}]" if item.badge else ""
                badge_space = (len(badge_str) + 1) if badge_str else 0

                avail_text = inner_w - len(pointer_str) - len(dot_str) - badge_space
                title_part = item.title
                sub_part = f" {item.subtitle}" if item.subtitle else ""

                if len(title_part) + len(sub_part) > avail_text:
                    overflow = (len(title_part) + len(sub_part)) - avail_text
                    if len(sub_part) > overflow + 2:
                        sub_part = sub_part[: -overflow - 1] + "…"
                    else:
                        sub_part = ""
                        if len(title_part) > avail_text:
                            title_part = title_part[: max(4, avail_text - 1)] + "…"

                fill_spaces = " " * max(0, avail_text - len(title_part) - len(sub_part))

                lines.append(("class:border", BORDER_V + " "))
                if is_active:
                    lines.append(("class:pointer_active", pointer_str))
                    lines.append(("class:item_selected_dot", dot_str))
                    lines.append(("class:item_selected", title_part))
                    if sub_part:
                        lines.append(("class:item_selected_dim", sub_part))
                    lines.append(("class:item_selected", fill_spaces))
                    if badge_str:
                        lines.append(("class:item_selected", " "))
                        lines.append(("class:item_selected_badge", badge_str))
                else:
                    lines.append(("class:card_bg", pointer_str))
                    if item.is_dot:
                        lines.append(("class:item_dot", dot_str))
                    elif item.is_checked:
                        lines.append(("class:item_check", dot_str))
                    else:
                        lines.append(("class:card_bg", dot_str))

                    lines.append(("class:item_normal", title_part))
                    if sub_part:
                        lines.append(("class:item_dim", sub_part))
                    lines.append(("class:card_bg", fill_spaces))
                    if badge_str:
                        lines.append(("class:card_bg", " "))
                        if badge_str.lower() in {"[connected]", "[active]"}:
                            lines.append(("class:item_badge_green", badge_str))
                        else:
                            lines.append(("class:item_badge", badge_str))

                lines.append(("class:border", " " + BORDER_V + "\n"))

            displayed_rows += 1

        if n_items == 0:
            no_text = "   No matching items found."
            lines.append(("class:border", BORDER_V + " "))
            lines.append(("class:no_matches", no_text))
            lines.append(("class:card_bg", " " * max(0, inner_w - len(no_text))))
            lines.append(("class:border", " " + BORDER_V + "\n"))
            displayed_rows += 1

        # Pad remaining rows to maintain perfectly stable card height
        needed_padding = max(0, visible_rows - displayed_rows)
        for _ in range(needed_padding):
            lines.append(("class:border", BORDER_V + " "))
            lines.append(("class:card_bg", " " * inner_w))
            lines.append(("class:border", " " + BORDER_V + "\n"))

        # 7. Separator below viewport (with Scroll Down indicator if more below)
        items_below = max(0, n_entries - (scroll_offset + visible_rows))
        if items_below > 0:
            hint = f" {ARROW_DOWN} {items_below} more below "
            rem = (card_width - 2) - 2 - len(hint)
            lines.append(("class:border", BORDER_LT + (BORDER_H * 2)))
            lines.append(("class:scroll_hint", hint))
            lines.append(("class:border", (BORDER_H * max(0, rem)) + BORDER_RT + "\n"))
        else:
            lines.append(("class:border", BORDER_LT + (BORDER_H * (card_width - 2)) + BORDER_RT + "\n"))

        # 8. Footer action toolbar
        k_nav = "[↑↓] "
        l_nav = "Move   "
        k_sel = "[Enter] "
        l_sel = "Select"
        if allow_ctrl_a:
            k_prov = "   [^A] "
            l_prov = footer_left
        else:
            k_prov = ""
            l_prov = ""

        k_esc = "[Esc] "
        l_esc = "Cancel "

        used_footer = (
            len(k_nav)
            + len(l_nav)
            + len(k_sel)
            + len(l_sel)
            + len(k_prov)
            + len(l_prov)
            + len(k_esc)
            + len(l_esc)
            + 1
        )
        footer_gap = max(1, inner_w - used_footer)

        lines.append(("class:border", BORDER_V + " "))
        lines.append(("class:card_bg", " "))
        lines.append(("class:footer_key", k_nav))
        lines.append(("class:footer_label", l_nav))
        lines.append(("class:footer_key", k_sel))
        lines.append(("class:footer_label", l_sel))
        if allow_ctrl_a:
            lines.append(("class:footer_key", k_prov))
            lines.append(("class:footer_label", l_prov))
        lines.append(("class:card_bg", " " * footer_gap))
        lines.append(("class:footer_key", k_esc))
        lines.append(("class:footer_label", l_esc))
        lines.append(("class:border", " " + BORDER_V + "\n"))

        # 9. Bottom border
        lines.append(("class:border", BORDER_BL + (BORDER_H * (card_width - 2)) + BORDER_BR + "\n"))

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
    # 1 top border + 1 title + 1 sep + 1 search + 1 sep + visible_rows + 1 sep + 1 footer + 1 bottom border
    total_card_height = visible_rows + 8

    card_window = Window(
        content=FormattedTextControl(render_content),
        width=Dimension.exact(card_width),
        height=Dimension.exact(total_card_height),
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
