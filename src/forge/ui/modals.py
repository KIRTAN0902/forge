"""Interactive In-App Modal Dialogs for Forge Workspace.

Implements native, non-blocking modal overlays for:
- ``/provider``: Multi-provider credential manager (Gemini, OpenRouter, OpenAI, Anthropic, DeepSeek, Groq, NVIDIA, Ollama, Custom)
- ``/models``: Multi-provider stage model selector matrix (cross-provider mixing for Anvil, Hammer, Temper)
- ``/resume``: Global cross-directory session history picker
- ``/usage``: Provider-grade API usage & cost telemetry dashboard
"""

from __future__ import annotations

import asyncio
import contextlib
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import (
    AnyFormattedText,
    StyleAndTextTuples,
)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Float,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.processors import DynamicProcessor, PasswordProcessor
from prompt_toolkit.widgets import Frame

from forge.config.session_store import SessionSummary, get_session_store
from forge.config.settings import (
    ForgeConfigManager,
    ProviderName,
)
from forge.config.usage_tracker import get_usage_tracker
from forge.ui.state import ForgeStage

if TYPE_CHECKING:  # pragma: no cover
    from forge.ui.cycle_controller import ForgeCycleController
    from forge.ui.split_app import ForgeSplitPaneApp


class ModalType(StrEnum):
    NONE = "none"
    PROVIDER = "provider"
    MODELS = "models"
    RESUME = "resume"
    USAGE = "usage"


PROVIDER_DISPLAY_NAMES: dict[ProviderName, str] = {
    ProviderName.GEMINI: "Google Gemini",
    ProviderName.OPENROUTER: "OpenRouter",
    ProviderName.OPENAI: "OpenAI",
    ProviderName.ANTHROPIC: "Anthropic Claude",
    ProviderName.DEEPSEEK: "DeepSeek",
    ProviderName.GROQ: "Groq",
    ProviderName.NVIDIA: "NVIDIA NIM",
    ProviderName.OLLAMA: "Ollama (Local)",
    ProviderName.CUSTOM: "Custom Provider",
}

STAGE_ROLES: list[tuple[str, str]] = [
    ("anvil", "1: Anvil (Orchestrator)"),
    ("hammer", "2: Hammer (Executor)"),
    ("temper", "3: Temper (QA)"),
    ("all", "4: All Stages"),
]


class ForgeModalManager:
    """Orchestrates interactive modal state, rendering, and key navigation."""

    def __init__(
        self,
        app: ForgeSplitPaneApp,
        config_manager: ForgeConfigManager | None = None,
        controller: ForgeCycleController | None = None,
    ) -> None:
        self.app = app
        self._config_manager = config_manager
        self._controller = controller
        self.current_modal: ModalType = ModalType.NONE
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Provider modal state
        self.provider_order: list[ProviderName] = [
            ProviderName.GEMINI,
            ProviderName.OPENROUTER,
            ProviderName.OPENAI,
            ProviderName.ANTHROPIC,
            ProviderName.DEEPSEEK,
            ProviderName.GROQ,
            ProviderName.NVIDIA,
            ProviderName.OLLAMA,
            ProviderName.CUSTOM,
        ]
        self.provider_index: int = 0
        self.input_mode: str | None = None  # None | 'api_key' | 'custom_url' | 'custom_key'
        self.active_provider_target: ProviderName | None = None
        self.status_message: str = ""

        # Models modal state
        self.target_role_index: int = 0  # 0: Anvil, 1: Hammer, 2: Temper, 3: All
        self.model_index: int = 0
        self.cached_aggregated_models: list[tuple[ProviderName, str]] = []
        self.model_search_query: str = ""

        # Resume modal state
        self.resume_index: int = 0
        self.cached_sessions: list[SessionSummary] = []

        # Usage modal state
        self.usage_window_index: int = 0  # 0: all, 1: today, 2: session
        self.usage_windows: list[str] = ["all", "today", "session"]

        # Text input buffer for modal forms
        self.input_buffer = Buffer(multiline=False)
        self.input_buffer.on_text_changed += self._on_input_text_changed

        # Build prompt_toolkit controls and windows
        self.modal_kb = KeyBindings()
        self.input_kb = KeyBindings()
        self._setup_keybindings()

        self.body_control = FormattedTextControl(
            text=self._render_body,
            key_bindings=self.modal_kb,
            show_cursor=False,
            focusable=True,
            modal=True,
        )
        self.body_window = Window(
            content=self.body_control,
            wrap_lines=False,
        )

        self.input_prompt_control = FormattedTextControl(
            text=self._render_input_prompt,
            show_cursor=False,
        )
        self.input_control = BufferControl(
            buffer=self.input_buffer,
            key_bindings=self.input_kb,
            input_processors=[
                DynamicProcessor(
                    lambda: PasswordProcessor()
                    if self.input_mode in {"api_key", "custom_key"}
                    else None
                )
            ],
        )
        self.input_field_window = Window(
            content=self.input_control,
            height=1,
            style="class:modal_input_field",
        )
        self.input_row = VSplit(
            [
                Window(content=self.input_prompt_control, dont_extend_width=True),
                self.input_field_window,
            ],
            height=1,
        )
        self.input_container = ConditionalContainer(
            content=HSplit(
                [
                    Window(height=1, char="─", style="class:border"),
                    self.input_row,
                ]
            ),
            filter=Condition(lambda: self.input_mode is not None),
        )

        self.modal_hsplit = HSplit(
            [
                self.body_window,
                self.input_container,
            ]
        )

        self.frame = Frame(
            body=self.modal_hsplit,
            title=self._render_frame_title,
            style="class:modal_frame",
        )

        self.container = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(text=""),
                dont_extend_height=True,
            ),
            filter=Condition(lambda: self.is_open),
        )

        # Centered floating dialog
        self.dialog_window = Window(
            content=FormattedTextControl(
                text=self._render_body,
                key_bindings=self.modal_kb,
                focusable=True,
                modal=True,
            ),
            wrap_lines=False,
        )

        self.modal_float = Float(
            content=ConditionalContainer(
                content=Frame(
                    body=HSplit([self.body_window, self.input_container]),
                    title=self._render_frame_title,
                    style="class:modal_dialog",
                    width=86,
                    height=24,
                ),
                filter=Condition(lambda: self.is_open),
            ),
        )

    @property
    def is_open(self) -> bool:
        return self.current_modal != ModalType.NONE

    @property
    def config_manager(self) -> ForgeConfigManager:
        if self.app.controller and hasattr(self.app.controller, "_config_manager"):
            mgr = self.app.controller._config_manager()
            if isinstance(mgr, ForgeConfigManager):
                return mgr
        if self.app.controller and hasattr(self.app.controller, "config_manager"):
            mgr = self.app.controller.config_manager
            if isinstance(mgr, ForgeConfigManager):
                return mgr
        if self._config_manager is None:
            self._config_manager = ForgeConfigManager()
        return self._config_manager

    def _create_task(self, coro: Any) -> asyncio.Task[Any] | None:
        """Spawn background task and keep strong reference to prevent GC."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if hasattr(coro, "close"):
                coro.close()
            return None
        task = loop.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    # -- modal open / close -------------------------------------------------------------

    def open_provider_modal(self) -> None:
        """Open the interactive Provider Connection Manager."""
        self.current_modal = ModalType.PROVIDER
        self.provider_index = 0
        self.input_mode = None
        self.status_message = ""
        self.input_buffer.text = ""
        self._focus_body()
        self._invalidate()

    def _on_input_text_changed(self, buf: Buffer) -> None:
        """Handle real-time search typing inside the input buffer."""
        if self.current_modal == ModalType.MODELS and self.input_mode == "model_search":
            self.model_search_query = buf.text
            self.model_index = 0
            self._invalidate()

    def open_models_modal(self, initial_search: str = "") -> None:
        """Open the interactive Multi-Provider Stage Model Selector Matrix."""
        self.current_modal = ModalType.MODELS
        self.model_index = 0
        self.model_search_query = initial_search.strip()
        self.status_message = ""
        self.refresh_aggregated_models()
        if initial_search:
            self.input_mode = "model_search"
            self.input_buffer.text = self.model_search_query
            self.input_buffer.cursor_position = len(self.input_buffer.text)
            self._focus_input()
        else:
            self.input_mode = None
            self.input_buffer.text = ""
            self._focus_body()
        self._invalidate()
        # Query live endpoints asynchronously to discover models
        self._create_task(self._query_live_models_async())

    def open_resume_modal(self) -> None:
        """Open the global session picker."""
        self.current_modal = ModalType.RESUME
        self.resume_index = 0
        self.cached_sessions = get_session_store().list_sessions()
        self.status_message = ""
        self._focus_body()
        self._invalidate()

    def open_usage_modal(self) -> None:
        """Open the API usage & telemetry dashboard modal."""
        self.current_modal = ModalType.USAGE
        self.status_message = ""
        self._focus_body()
        self._invalidate()

    def close_modal(self) -> None:
        """Close any open modal and restore focus to the input box."""
        self.current_modal = ModalType.NONE
        self.input_mode = None
        self.status_message = ""
        self.input_buffer.text = ""
        self.model_search_query = ""
        with contextlib.suppress(Exception):
            self.app.application.layout.focus(self.app.input_window)
        self._invalidate()

    def _focus_body(self) -> None:
        with contextlib.suppress(Exception):
            self.app.application.layout.focus(self.body_window)

    def _focus_input(self) -> None:
        with contextlib.suppress(Exception):
            self.app.application.layout.focus(self.input_field_window)

    def _invalidate(self) -> None:
        with contextlib.suppress(Exception):
            self.app.application.invalidate()

    # -- model aggregation & search -----------------------------------------------------

    @property
    def filtered_models(self) -> list[tuple[ProviderName, str]]:
        """Return cached aggregated models filtered by model_search_query."""
        if not self.model_search_query:
            return self.cached_aggregated_models
        terms = [t for t in self.model_search_query.lower().split() if t]
        if not terms:
            return self.cached_aggregated_models
        results: list[tuple[ProviderName, str]] = []
        for prov, name in self.cached_aggregated_models:
            target = f"{prov.value}/{name}".lower()
            if all(t in target for t in terms):
                results.append((prov, name))
        return results

    def refresh_aggregated_models(self) -> None:
        """Combine all models from every connected provider."""
        mgr = self.config_manager
        aggregated = mgr.get_aggregated_models()
        if not aggregated:
            # Fall back to default curated models of all providers if none connected yet
            for p in self.provider_order:
                cfg = mgr.provider(p)
                for m in cfg.available_models:
                    aggregated.append((p, m))
        self.cached_aggregated_models = aggregated
        models = self.filtered_models
        if self.model_index >= len(models):
            self.model_index = max(0, len(models) - 1)

    async def _query_live_models_async(self) -> None:
        """Asynchronously query live endpoints of connected providers."""
        from forge.providers.client import UnifiedLLMClient

        mgr = self.config_manager
        client = UnifiedLLMClient(
            api_key_resolver=mgr.api_key_for,
            base_url_resolver=lambda p: mgr.provider(p).base_url,
        )
        updated = False
        for p in mgr.get_connected_providers():
            try:
                models = await client.list_live_models(p)
                if models:
                    cfg = mgr.provider(p)
                    for m in models:
                        if m not in cfg.available_models:
                            cfg.available_models.append(m)
                            updated = True
            except Exception:
                continue
        if updated:
            self.refresh_aggregated_models()
            self._invalidate()

    # -- keybindings -------------------------------------------------------------------

    def _setup_keybindings(self) -> None:
        kb = self.modal_kb

        @kb.add("up")
        def _up(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.PROVIDER:
                self.provider_index = (self.provider_index - 1) % len(self.provider_order)
            elif self.current_modal == ModalType.MODELS:
                count = max(1, len(self.filtered_models))
                self.model_index = (self.model_index - 1) % count
            elif self.current_modal == ModalType.RESUME:
                count = max(1, len(self.cached_sessions))
                self.resume_index = (self.resume_index - 1) % count
            self._invalidate()

        @kb.add("down")
        def _down(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.PROVIDER:
                self.provider_index = (self.provider_index + 1) % len(self.provider_order)
            elif self.current_modal == ModalType.MODELS:
                count = max(1, len(self.filtered_models))
                self.model_index = (self.model_index + 1) % count
            elif self.current_modal == ModalType.RESUME:
                count = max(1, len(self.cached_sessions))
                self.resume_index = (self.resume_index + 1) % count
            self._invalidate()

        @kb.add("left")
        def _left(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS:
                self.target_role_index = (self.target_role_index - 1) % len(STAGE_ROLES)
                self._invalidate()
            elif self.current_modal == ModalType.USAGE:
                self.usage_window_index = (self.usage_window_index - 1) % len(self.usage_windows)
                self._invalidate()

        @kb.add("right")
        def _right(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS:
                self.target_role_index = (self.target_role_index + 1) % len(STAGE_ROLES)
                self._invalidate()
            elif self.current_modal == ModalType.USAGE:
                self.usage_window_index = (self.usage_window_index + 1) % len(self.usage_windows)
                self._invalidate()

        @kb.add("tab")
        def _tab(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS:
                self.target_role_index = (self.target_role_index + 1) % len(STAGE_ROLES)
                self._invalidate()
            elif self.current_modal == ModalType.USAGE:
                self.usage_window_index = (self.usage_window_index + 1) % len(self.usage_windows)
                self._invalidate()

        for idx, key_num in enumerate(["1", "2", "3", "4"]):

            def _bind_num(i: int = idx, k: str = key_num) -> None:
                @kb.add(k)
                def _handle_num(event: KeyPressEvent) -> None:
                    if self.current_modal == ModalType.MODELS and i < len(STAGE_ROLES):
                        self.target_role_index = i
                        self._invalidate()
                    elif self.current_modal == ModalType.USAGE and i < len(self.usage_windows):
                        self.usage_window_index = i
                        self._invalidate()

            _bind_num()

        @kb.add("/")
        @kb.add("s")
        @kb.add("c-f")
        def _start_search(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS:
                self.input_mode = "model_search"
                self.input_buffer.text = self.model_search_query
                self.input_buffer.cursor_position = len(self.input_buffer.text)
                self._focus_input()
                self._invalidate()

        @kb.add("c")
        def _clear_search(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS and self.model_search_query:
                self.model_search_query = ""
                self.input_buffer.text = ""
                self.model_index = 0
                self.status_message = "Search filter cleared."
                self._invalidate()

        @kb.add("enter")
        def _enter(event: KeyPressEvent) -> None:
            self._handle_enter()

        @kb.add("delete")
        @kb.add("d")
        def _delete(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.PROVIDER:
                prov = self.provider_order[self.provider_index]
                self.config_manager.disconnect_provider(prov)
                self.status_message = f"Disconnected {PROVIDER_DISPLAY_NAMES[prov]}."
                if self.app.controller:
                    self._create_task(self.app.controller.refresh_models())
                self._invalidate()
            elif self.current_modal == ModalType.MODELS and self.model_search_query:
                self.model_search_query = ""
                self.input_buffer.text = ""
                self.model_index = 0
                self.status_message = "Search filter cleared."
                self._invalidate()

        @kb.add("escape")
        def _esc(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS and self.model_search_query:
                self.model_search_query = ""
                self.input_buffer.text = ""
                self.model_index = 0
                self.status_message = "Search filter cleared."
                self._invalidate()
            else:
                self.close_modal()

        @kb.add("c-c")
        def _ctrl_c(event: KeyPressEvent) -> None:
            self.close_modal()

        # Keybindings for input field
        in_kb = self.input_kb

        @in_kb.add("up")
        def _in_up(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS:
                count = max(1, len(self.filtered_models))
                self.model_index = (self.model_index - 1) % count
                self._invalidate()

        @in_kb.add("down")
        def _in_down(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS:
                count = max(1, len(self.filtered_models))
                self.model_index = (self.model_index + 1) % count
                self._invalidate()

        @in_kb.add("tab")
        def _in_tab(event: KeyPressEvent) -> None:
            if self.current_modal == ModalType.MODELS:
                self.target_role_index = (self.target_role_index + 1) % len(STAGE_ROLES)
                self._invalidate()

        @in_kb.add("enter")
        def _in_enter(event: KeyPressEvent) -> None:
            self._handle_input_enter()

        @in_kb.add("escape")
        def _in_esc(event: KeyPressEvent) -> None:
            self.input_mode = None
            self._focus_body()
            self._invalidate()

        @in_kb.add("c-c")
        def _in_ctrl_c(event: KeyPressEvent) -> None:
            self.close_modal()

    def _handle_enter(self) -> None:
        """Handle Enter key pressed on modal body."""
        if self.current_modal == ModalType.PROVIDER:
            prov = self.provider_order[self.provider_index]
            self.active_provider_target = prov
            if prov == ProviderName.CUSTOM:
                self.input_mode = "custom_url"
                self.input_buffer.text = self.config_manager.provider(prov).base_url
            elif prov == ProviderName.OLLAMA:
                self.input_mode = "ollama_url"
                self.input_buffer.text = self.config_manager.provider(prov).base_url
            else:
                self.input_mode = "api_key"
                self.input_buffer.text = ""
            self._focus_input()
            self._invalidate()

        elif self.current_modal == ModalType.MODELS:
            models = self.filtered_models
            if not models:
                return
            if self.model_index >= len(models):
                self.model_index = max(0, len(models) - 1)
            prov, model_name = models[self.model_index]
            role_target, _ = STAGE_ROLES[self.target_role_index]
            spec = f"{prov.value}/{model_name}"

            # Persist immediately to config.json
            self.config_manager.set_stage_model(role_target, spec)
            self.status_message = f"[OK] Bound {role_target.upper()} -> {spec} (Live Active)"

            # Live activation without restart
            if self.app.controller:
                self._create_task(self._activate_model_live(role_target, prov.value, model_name))
            self._invalidate()

        elif self.current_modal == ModalType.RESUME:
            if not self.cached_sessions:
                return
            session = self.cached_sessions[self.resume_index]
            get_session_store().restore_session(session.session_id, self.app.state)
            self.close_modal()

    async def _activate_model_live(self, target: str, provider: str, model: str) -> None:
        ctrl = self.app.controller
        if not ctrl:
            return
        if target == "all":
            await ctrl._apply_model_spec("all", f"{provider}/{model}")
        else:
            stage = ForgeStage.ANVIL if target == "anvil" else (ForgeStage.HAMMER if target == "hammer" else ForgeStage.TEMPER)
            await ctrl.swap_stage_model(stage, provider, model)
        await ctrl.refresh_models()
        self._invalidate()

    def _handle_input_enter(self) -> None:
        """Handle Enter key pressed inside the modal text input field."""
        if self.input_mode == "model_search":
            models = self.filtered_models
            if models:
                if self.model_index >= len(models):
                    self.model_index = max(0, len(models) - 1)
                prov, model_name = models[self.model_index]
                role_target, _ = STAGE_ROLES[self.target_role_index]
                spec = f"{prov.value}/{model_name}"

                self.config_manager.set_stage_model(role_target, spec)
                self.status_message = f"[OK] Bound {role_target.upper()} -> {spec} (Live Active)"

                if self.app.controller:
                    self._create_task(self._activate_model_live(role_target, prov.value, model_name))
            self.input_mode = None
            self._focus_body()
            self._invalidate()
            return

        text = self.input_buffer.text.strip()
        target_prov = self.active_provider_target

        if self.input_mode == "api_key" and target_prov is not None:
            if text:
                self.config_manager.set_api_key(target_prov, text)
                self.status_message = f"[OK] Saved API key for {PROVIDER_DISPLAY_NAMES[target_prov]}."
                if self.app.controller:
                    self._create_task(self.app.controller.refresh_models())
            self.input_mode = None
            self.input_buffer.text = ""
            self._focus_body()
            self._invalidate()

        elif self.input_mode == "custom_url" and target_prov is not None:
            if text:
                self.config_manager.set_base_url(target_prov, text)
            # Switch to prompting for custom API key
            self.input_mode = "custom_key"
            self.input_buffer.text = ""
            self._focus_input()
            self._invalidate()

        elif self.input_mode == "custom_key" and target_prov is not None:
            if text:
                self.config_manager.set_api_key(target_prov, text)
            self.status_message = f"[OK] Custom provider configured at {self.config_manager.provider(target_prov).base_url}."
            if self.app.controller:
                self._create_task(self.app.controller.refresh_models())
            self.input_mode = None
            self.input_buffer.text = ""
            self._focus_body()
            self._invalidate()

        elif self.input_mode == "ollama_url" and target_prov is not None:
            if text:
                self.config_manager.set_base_url(target_prov, text)
            self.status_message = f"[OK] Ollama endpoint updated to {text}."
            if self.app.controller:
                self._create_task(self.app.controller.refresh_models())
            self.input_mode = None
            self.input_buffer.text = ""
            self._focus_body()
            self._invalidate()

    # -- renderers ---------------------------------------------------------------------

    def _render_frame_title(self) -> AnyFormattedText:
        if self.current_modal == ModalType.PROVIDER:
            return [("class:modal_title", " FORGE // PROVIDER CONNECTION MANAGER ")]
        if self.current_modal == ModalType.MODELS:
            return [("class:modal_title", " FORGE // MULTI-PROVIDER STAGE MODEL MATRIX ")]
        if self.current_modal == ModalType.RESUME:
            return [("class:modal_title", " FORGE // GLOBAL CONVERSATION RESUME ")]
        if self.current_modal == ModalType.USAGE:
            win = self.usage_windows[self.usage_window_index].upper()
            return [("class:modal_title", f" FORGE // API USAGE & COST TELEMETRY [{win}] ")]
        return [("class:modal_title", " FORGE ")]

    def _render_input_prompt(self) -> AnyFormattedText:
        prov_name = PROVIDER_DISPLAY_NAMES.get(self.active_provider_target or ProviderName.GEMINI, "")
        if self.input_mode == "api_key":
            return [("class:modal_prompt", f" Enter API Key (masked) for {prov_name}: ")]
        if self.input_mode == "custom_url":
            return [("class:modal_prompt", " Base URI/URL (e.g. http://localhost:8000/v1): ")]
        if self.input_mode == "custom_key":
            return [("class:modal_prompt", " API Key (optional for local): ")]
        if self.input_mode == "ollama_url":
            return [("class:modal_prompt", " Ollama Base URL: ")]
        if self.input_mode == "model_search":
            return [("class:modal_prompt", " [SEARCH] Search / Filter models: ")]
        return [("class:modal_prompt", " Input: ")]

    def _render_body(self) -> StyleAndTextTuples:
        if self.current_modal == ModalType.PROVIDER:
            return self._render_provider_modal()
        if self.current_modal == ModalType.MODELS:
            return self._render_models_modal()
        if self.current_modal == ModalType.RESUME:
            return self._render_resume_modal()
        if self.current_modal == ModalType.USAGE:
            return self._render_usage_modal()
        return []

    def _render_provider_modal(self) -> StyleAndTextTuples:
        frags: StyleAndTextTuples = []
        frags.append(("class:modal_hint", " Connect providers with zero restarts. Press [Enter] to connect/edit, [d] to disconnect.\n"))
        frags.append(("class:border", "─" * 82 + "\n\n"))

        mgr = self.config_manager
        for idx, prov in enumerate(self.provider_order):
            is_selected = idx == self.provider_index
            is_conn = mgr.is_connected(prov)
            cfg = mgr.provider(prov)
            name = PROVIDER_DISPLAY_NAMES[prov]

            marker = ">" if is_selected else " "
            check = "[✓]" if is_conn else "[ ]"

            if is_conn:
                if prov == ProviderName.OLLAMA:
                    status_text = f"(Connected: {cfg.base_url})"
                elif prov == ProviderName.CUSTOM:
                    key_part = f", Key: {cfg.masked_key()}" if cfg.api_key else ""
                    status_text = f"(Connected: {cfg.base_url}{key_part})"
                else:
                    status_text = f"(Connected: {cfg.masked_key()})"
            else:
                if prov == ProviderName.OLLAMA:
                    status_text = "(Auto-detected / Localhost)"
                elif prov == ProviderName.CUSTOM:
                    status_text = "(Requires Base URL + API Key)"
                else:
                    status_text = "(Not connected)"

            line_style = "class:modal_item_selected" if is_selected else "class:modal_item"
            check_style = "class:modal_check_on" if is_conn else "class:modal_check_off"

            frags.append((line_style, f" {marker} "))
            frags.append((check_style, f"{check} "))
            frags.append((line_style, f"{name:<20} {status_text:<48}\n"))

        frags.append(("", "\n"))
        if self.status_message:
            frags.append(("class:modal_success", f" {self.status_message}\n"))
        else:
            frags.append(("class:border", "─" * 82 + "\n"))

        frags.append(
            ("class:modal_footer", " [↑/↓] Navigate  │  [Enter] Connect  │  [d/Del] Disconnect  │  [Esc] Done\n")
        )
        return frags

    def _render_models_modal(self) -> StyleAndTextTuples:
        frags: StyleAndTextTuples = []
        mgr = self.config_manager

        # Current live bindings
        b_anvil = mgr.binding_for("anvil")
        b_hammer = mgr.binding_for("hammer")
        b_temper = mgr.binding_for("temper")

        s_anvil = b_anvil.spec_string if b_anvil else "(none)"
        s_hammer = b_hammer.spec_string if b_hammer else "(none)"
        s_temper = b_temper.spec_string if b_temper else "(none)"

        frags.append(("class:modal_subtitle", " Current Live Stage Bindings:\n"))
        frags.append(("class:anvil", f"   • Anvil  (Orchestrator) : {s_anvil}\n"))
        frags.append(("class:hammer", f"   • Hammer (Executor)     : {s_hammer}\n"))
        frags.append(("class:temper", f"   • Temper (QA)           : {s_temper}\n\n"))

        # Target Role Selector Tabs
        frags.append(("class:modal_hint", " Target Role:  "))
        for idx, (_role_id, role_label) in enumerate(STAGE_ROLES):
            is_active_tab = idx == self.target_role_index
            if is_active_tab:
                frags.append(("class:modal_tab_active", f" [ {role_label} ] "))
            else:
                frags.append(("class:modal_tab_inactive", f"   {role_label}   "))
        frags.append(("", "\n"))
        frags.append(("class:border", "─" * 82 + "\n"))

        # Search Bar Display
        total_count = len(self.cached_aggregated_models)
        models = self.filtered_models
        filtered_count = len(models)

        if self.input_mode == "model_search":
            search_badge = " [[SEARCH] SEARCHING] "
            search_text = f"Query: \"{self.model_search_query}\" ({filtered_count}/{total_count} models) - type query below"
            frags.append(("class:modal_tab_active", search_badge))
            frags.append(("class:modal_prompt", f" {search_text}\n"))
        elif self.model_search_query:
            search_badge = " [[SEARCH] FILTERED] "
            search_text = f"Query: \"{self.model_search_query}\" ({filtered_count}/{total_count} models) - [/] to edit, [c] to clear"
            frags.append(("class:modal_tab_active", search_badge))
            frags.append(("class:modal_prompt", f" {search_text}\n"))
        else:
            frags.append(
                (
                    "class:modal_hint",
                    f" Models ({total_count} available)  │  Press [/] or [s] to search & filter models\n",
                )
            )
        frags.append(("class:border", "─" * 82 + "\n"))

        if not models:
            if self.model_search_query:
                frags.append(
                    (
                        "class:modal_hint",
                        f"   (No models matching '{self.model_search_query}'. Press [c] or [Esc] to clear search)\n",
                    )
                )
            else:
                frags.append(("class:modal_hint", "   (No connected providers or models found)\n"))
        else:
            visible_window = 6
            start = max(0, min(self.model_index - 2, len(models) - visible_window))
            end = min(len(models), start + visible_window)

            for idx in range(start, end):
                prov, m_name = models[idx]
                is_sel = idx == self.model_index
                marker = ">" if is_sel else " "
                badge = f"[{prov.value}]"
                line_style = "class:modal_item_selected" if is_sel else "class:modal_item"
                badge_style = (
                    f"class:{prov.value}"
                    if prov.value in {"anvil", "hammer", "temper"}
                    else "class:modal_badge"
                )

                frags.append((line_style, f" {marker} "))
                frags.append((badge_style, f"{badge:<14} "))
                frags.append((line_style, f"{m_name}\n"))

        frags.append(("", "\n"))
        if self.status_message:
            frags.append(("class:modal_success", f" {self.status_message}\n"))
        else:
            frags.append(("class:border", "─" * 82 + "\n"))

        if self.input_mode == "model_search":
            footer = " [Type] Filter Query │ [↑/↓] Select │ [Tab] Target Role │ [Enter] Bind │ [Esc] Finish Search\n"
        elif self.model_search_query:
            footer = " [←/→/Tab/1-4] Role │ [↑/↓] Select │ [/] Edit Search │ [c] Clear │ [Enter] Bind │ [Esc] Done\n"
        else:
            footer = " [←/→/Tab/1-4] Role │ [↑/↓] Select │ [/] Search Models │ [Enter] Bind │ [Esc] Done\n"

        frags.append(("class:modal_footer", footer))
        return frags

    def _render_resume_modal(self) -> StyleAndTextTuples:
        frags: StyleAndTextTuples = []
        frags.append(("class:modal_hint", " Resume any past conversation session across your entire system.\n"))
        frags.append(("class:border", "─" * 82 + "\n"))

        sessions = self.cached_sessions
        if not sessions:
            frags.append(("class:modal_hint", "  (No past sessions found in ~/.forge/sessions/)\n"))
        else:
            visible_window = 10
            start = max(0, min(self.resume_index - 4, len(sessions) - visible_window))
            end = min(len(sessions), start + visible_window)

            for idx in range(start, end):
                s = sessions[idx]
                is_sel = idx == self.resume_index
                marker = ">" if is_sel else " "
                line_style = "class:modal_item_selected" if is_sel else "class:modal_item"
                ts = s.updated_at[:16].replace("T", " ") if s.updated_at else ""

                frags.append((line_style, f" {marker} [{s.session_id[:8]}] {ts} │ {s.title[:28]:<28} │ {s.total_tokens:>6,} tok │ Cyc:{s.cycle_count}\n"))

        frags.append(("", "\n"))
        frags.append(("class:border", "─" * 82 + "\n"))
        frags.append(("class:modal_footer", " [↑/↓] Navigate  │  [Enter] Resume Session  │  [Esc] Cancel\n"))
        return frags

    def _render_usage_modal(self) -> StyleAndTextTuples:
        frags: StyleAndTextTuples = []
        win = self.usage_windows[self.usage_window_index]

        # Time window selector tabs
        frags.append(("class:modal_hint", " Time Window:  "))
        for idx, w_name in enumerate(self.usage_windows):
            is_active = idx == self.usage_window_index
            lbl = w_name.title()
            if is_active:
                frags.append(("class:modal_tab_active", f" [ {idx + 1}: {lbl} ] "))
            else:
                frags.append(("class:modal_tab_inactive", f"   {idx + 1}: {lbl}   "))
        frags.append(("", "\n"))
        frags.append(("class:border", "─" * 82 + "\n"))

        # Render the controller's live three-part accounting when available.
        controller = self.app.controller
        renderer = getattr(controller, "render_usage_report", None)
        if callable(renderer):
            text = str(renderer())
        else:
            tracker = get_usage_tracker()
            session_id = getattr(controller, "session_id", None)
            text = tracker.render_dashboard_text(session_id=session_id, time_window=win)

        for line in text.splitlines()[:24]:
            frags.append(("class:modal_item", f" {line}\n"))

        frags.append(("class:border", "─" * 82 + "\n"))
        frags.append(("class:modal_footer", " [Tab/←/→/1-3] Switch Filter  │  [Esc] Close\n"))
        return frags
