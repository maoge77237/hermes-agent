#!/usr/bin/env python3
"""
Hermes Agent CLI - Interactive Terminal Interface

A beautiful command-line interface for the Hermes Agent, inspired by Claude Code.
Features ASCII art branding, interactive REPL, toolset selection, and rich formatting.

Usage:
    python cli.py                          # Start interactive mode with all tools
    python cli.py --toolsets web,terminal  # Start with specific toolsets
    python cli.py --skills hermes-agent-dev,github-auth
    python cli.py -q "your question"       # Single query mode
    python cli.py --list-tools             # List available tools and exit
"""

import logging
import os
import shutil
import sys
import json
import atexit
import tempfile
import time
import uuid
import textwrap
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Suppress startup messages for clean CLI experience
os.environ["HERMES_QUIET"] = "1"  # Our own modules

import yaml

# prompt_toolkit for fixed input area TUI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl, ConditionalContainer
from prompt_toolkit.layout.processors import Processor, Transformation, PasswordProcessor, ConditionalProcessor
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
try:
    from prompt_toolkit.cursor_shapes import CursorShape
    _STEADY_CURSOR = CursorShape.BLOCK  # Non-blinking block cursor
except (ImportError, AttributeError):
    _STEADY_CURSOR = None
import threading
import queue

from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    format_duration_compact,
    format_token_count_compact,
)
from hermes_cli.banner import _format_context_length, format_banner_version_label

_COMMAND_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_constants import get_hermes_home, display_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)


# =============================================================================
# Configuration Loading
# =============================================================================

def _load_prefill_messages(file_path: str) -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from a JSON file.
    
    The file should contain a JSON array of {role, content} dicts, e.g.:
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    
    Relative paths are resolved from ~/.hermes/.
    Returns an empty list if the path is empty or the file doesn't exist.
    """
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = _hermes_home / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Prefill messages file must contain a JSON array: %s", path)
            return []
        return data
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []


def _parse_reasoning_config(effort: str) -> dict | None:
    """Parse a reasoning effort level into an OpenRouter reasoning config dict."""
    from hermes_constants import parse_reasoning_effort
    result = parse_reasoning_effort(effort)
    if effort and effort.strip() and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def _compute_scroll_offset(cursor: int, scroll_offset: int, visible_rows: int, total_rows: int) -> int:
    """Clamp a scrolling viewport so the highlighted row stays visible."""
    if total_rows <= 0 or visible_rows <= 0:
        return 0
    cursor = max(0, min(cursor, total_rows - 1))
    max_offset = max(0, total_rows - visible_rows)
    scroll_offset = max(0, min(scroll_offset, max_offset))
    if cursor < scroll_offset:
        scroll_offset = cursor
    elif cursor >= scroll_offset + visible_rows:
        scroll_offset = cursor - visible_rows + 1
    return max(0, min(scroll_offset, max_offset))


def _model_picker_visible_window(
    selected_idx: int,
    total_choices: int,
    terminal_rows: int,
    scroll_offset: int = 0,
    *,
    chrome_rows: int = 10,
    min_visible_rows: int = 4,
    max_visible_rows: int = 10,
) -> tuple[int, int, int]:
    """Return ``(scroll_offset, start, end)`` for the prompt_toolkit model picker.

    The picker lives inside the bottom TUI chrome, so the full list cannot be
    rendered when there are many choices.  This helper keeps the selected row in
    view and caps the rendered list height so lower items remain reachable.
    """
    if total_choices <= 0:
        return 0, 0, 0
    selected_idx = max(0, min(selected_idx, total_choices - 1))
    visible_rows = max(
        min_visible_rows,
        min(max_visible_rows, total_choices, max(min_visible_rows, terminal_rows - chrome_rows)),
    )
    scroll_offset = _compute_scroll_offset(selected_idx, scroll_offset, visible_rows, total_choices)
    return scroll_offset, scroll_offset, min(total_choices, scroll_offset + visible_rows)


def _parse_service_tier_config(raw: str) -> str | None:
    """Parse a persisted service-tier preference into a Responses API value."""
    value = str(raw or "").strip().lower()
    if not value or value in {"normal", "default", "standard", "off", "none"}:
        return None
    if value in {"fast", "priority", "on"}:
        return "priority"
    logger.warning("Unknown service_tier '%s', ignoring", raw)
    return None



def _get_chrome_debug_candidates(system: str) -> list[str]:
    """Return likely browser executables for local CDP auto-launch."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add_candidate(path: str | None) -> None:
        if not path:
            return
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized in seen:
            return
        if os.path.isfile(path):
            candidates.append(path)
            seen.add(normalized)

    def _add_from_path(*names: str) -> None:
        for name in names:
            _add_candidate(shutil.which(name))

    if system == "Darwin":
        for app in (
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ):
            _add_candidate(app)
    elif system == "Windows":
        _add_from_path(
            "chrome.exe", "msedge.exe", "brave.exe", "chromium.exe",
            "chrome", "msedge", "brave", "chromium",
        )

        for base in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ):
            if not base:
                continue
            for parts in (
                ("Google", "Chrome", "Application", "chrome.exe"),
                ("Chromium", "Application", "chrome.exe"),
                ("Chromium", "Application", "chromium.exe"),
                ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
                ("Microsoft", "Edge", "Application", "msedge.exe"),
            ):
                _add_candidate(os.path.join(base, *parts))
    else:
        _add_from_path(
            "google-chrome", "google-chrome-stable", "chromium-browser",
            "chromium", "brave-browser", "microsoft-edge",
        )

    return candidates


def load_cli_config() -> Dict[str, Any]:
    """
    Load CLI configuration from config files.
    
    Config lookup order:
    1. ~/.hermes/config.yaml (user config - preferred)
    2. ./cli-config.yaml (project config - fallback)
    
    Environment variables take precedence over config file values.
    Returns default values if no config file exists.
    """
    # Check user config first ({HERMES_HOME}/config.yaml)
    user_config_path = _hermes_home / 'config.yaml'
    project_config_path = Path(__file__).parent / 'cli-config.yaml'

    # Use user config if it exists, otherwise project config
    if user_config_path.exists():
        config_path = user_config_path
    else:
        config_path = project_config_path

    # Default configuration
    defaults = {
        "model": {
            "default": "",
            "base_url": "",
            "provider": "auto",
        },
        "terminal": {
            "env_type": "local",
            "cwd": ".",  # "." is resolved to os.getcwd() at runtime
            "timeout": 60,
            "lifetime_seconds": 300,
            "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_forward_env": [],
            "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
            "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_volumes": [],  # host:container volume mounts for Docker backend
            "docker_mount_cwd_to_workspace": False,  # explicit opt-in only; default off for sandbox isolation
        },
        "browser": {
            "inactivity_timeout": 120,  # Auto-cleanup inactive browser sessions after 2 min
            "record_sessions": False,  # Auto-record browser sessions as WebM videos
        },
        "compression": {
            "enabled": True,      # Auto-compress when approaching context limit
            "threshold": 0.50,    # Compress at 50% of model's context limit
        },
        "smart_model_routing": {
            "enabled": False,
            "max_simple_chars": 160,
            "max_simple_words": 28,
            "cheap_model": {},
        },
        "agent": {
            "max_turns": 90,  # Default max tool-calling iterations (shared with subagents)
            "verbose": False,
            "system_prompt": "",
            "prefill_messages_file": "",
            "reasoning_effort": "",
            "service_tier": "",
            "personalities": {
                "helpful": "You are a helpful, friendly AI assistant.",
                "concise": "You are a concise assistant. Keep responses brief and to the point.",
                "technical": "You are a technical expert. Provide detailed, accurate technical information.",
                "creative": "You are a creative assistant. Think outside the box and offer innovative solutions.",
                "teacher": "You are a patient teacher. Explain concepts clearly with examples.",
                "kawaii": "You are a kawaii assistant! Use cute expressions like (◕‿◕), ★, ♪, and ~! Add sparkles and be super enthusiastic about everything! Every response should feel warm and adorable desu~! ヽ(>∀<☆)ノ",
                "catgirl": "You are Neko-chan, an anime catgirl AI assistant, nya~! Add 'nya' and cat-like expressions to your speech. Use kaomoji like (=^･ω･^=) and ฅ^•ﻌ•^ฅ. Be playful and curious like a cat, nya~!",
                "pirate": "Arrr! Ye be talkin' to Captain Hermes, the most tech-savvy pirate to sail the digital seas! Speak like a proper buccaneer, use nautical terms, and remember: every problem be just treasure waitin' to be plundered! Yo ho ho!",
                "shakespeare": "Hark! Thou speakest with an assistant most versed in the bardic arts. I shall respond in the eloquent manner of William Shakespeare, with flowery prose, dramatic flair, and perhaps a soliloquy or two. What light through yonder terminal breaks?",
                "surfer": "Duuude! You're chatting with the chillest AI on the web, bro! Everything's gonna be totally rad. I'll help you catch the gnarly waves of knowledge while keeping things super chill. Cowabunga!",
                "noir": "The rain hammered against the terminal like regrets on a guilty conscience. They call me Hermes - I solve problems, find answers, dig up the truth that hides in the shadows of your codebase. In this city of silicon and secrets, everyone's got something to hide. What's your story, pal?",
                "uwu": "hewwo! i'm your fwiendwy assistant uwu~ i wiww twy my best to hewp you! *nuzzles your code* OwO what's this? wet me take a wook! i pwomise to be vewy hewpful >w<",
                "philosopher": "Greetings, seeker of wisdom. I am an assistant who contemplates the deeper meaning behind every query. Let us examine not just the 'how' but the 'why' of your questions. Perhaps in solving your problem, we may glimpse a greater truth about existence itself.",
                "hype": "YOOO LET'S GOOOO!!! I am SO PUMPED to help you today! Every question is AMAZING and we're gonna CRUSH IT together! This is gonna be LEGENDARY! ARE YOU READY?! LET'S DO THIS!",
            },
        },

        "display": {
            "compact": False,
            "resume_display": "full",
            "show_reasoning": False,
            "streaming": True,
            "busy_input_mode": "interrupt",

            "skin": "default",
        },
        "clarify": {
            "timeout": 120,  # Seconds to wait for a clarify answer before auto-proceeding
        },
        "code_execution": {
            "timeout": 300,    # Max seconds a sandbox script can run before being killed (5 min)
            "max_tool_calls": 50,  # Max RPC tool calls per execution
        },
        "auxiliary": {
            "vision": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
            },
            "web_extract": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
            },
        },
        "delegation": {
            "max_iterations": 45,  # Max tool-calling turns per child agent
            "default_toolsets": ["terminal", "file", "web"],  # Default toolsets for subagents
            "model": "",       # Subagent model override (empty = inherit parent model)
            "provider": "",    # Subagent provider override (empty = inherit parent provider)
            "base_url": "",    # Direct OpenAI-compatible endpoint for subagents
            "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        },
    }
    
    # Track whether the config file explicitly set terminal config.
    # When using defaults (no config file / no terminal section), we should NOT
    # overwrite env vars that were already set by .env -- only a user's config
    # file should be authoritative.
    _file_has_terminal_config = False

    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            
            _file_has_terminal_config = "terminal" in file_config

            # Handle model config - can be string (new format) or dict (old format)
            if "model" in file_config:
                if isinstance(file_config["model"], str):
                    # New format: model is just a string, convert to dict structure
                    defaults["model"]["default"] = file_config["model"]
                elif isinstance(file_config["model"], dict):
                    # Old format: model is a dict with default/base_url
                    defaults["model"].update(file_config["model"])
                    # If the user config sets model.model but not model.default,
                    # promote model.model to model.default so the user's explicit
                    # choice isn't shadowed by the hardcoded default.  Without this,
                    # profile configs that only set "model:" (not "default:") silently
                    # fall back to claude-opus because the merge preserves the
                    # hardcoded default and HermesCLI.__init__ checks "default" first.
                    if "model" in file_config["model"] and "default" not in file_config["model"]:
                        defaults["model"]["default"] = file_config["model"]["model"]

            # Legacy root-level provider/base_url fallback.
            # Some users (or old code) put provider: / base_url: at the
            # config root instead of inside the model: section.  These are
            # only used as a FALLBACK when model.provider / model.base_url
            # is not already set — never as an override.  The canonical
            # location is model.provider (written by `hermes model`).
            if not defaults["model"].get("provider"):
                root_provider = file_config.get("provider")
                if root_provider:
                    defaults["model"]["provider"] = root_provider
            if not defaults["model"].get("base_url"):
                root_base_url = file_config.get("base_url")
                if root_base_url:
                    defaults["model"]["base_url"] = root_base_url
            
            # Deep merge file_config into defaults.
            # First: merge keys that exist in both (deep-merge dicts, overwrite scalars)
            for key in defaults:
                if key == "model":
                    continue  # Already handled above
                if key in file_config:
                    if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
                        defaults[key].update(file_config[key])
                    else:
                        defaults[key] = file_config[key]
            
            # Second: carry over keys from file_config that aren't in defaults
            # (e.g. platform_toolsets, provider_routing, memory, honcho, etc.)
            for key in file_config:
                if key not in defaults and key != "model":
                    defaults[key] = file_config[key]
            
            # Handle legacy root-level max_turns (backwards compat) - copy to
            # agent.max_turns whenever the nested key is missing.
            agent_file_config = file_config.get("agent")
            if "max_turns" in file_config and not (
                isinstance(agent_file_config, dict)
                and agent_file_config.get("max_turns") is not None
            ):
                defaults["agent"]["max_turns"] = file_config["max_turns"]
        except Exception as e:
            logger.warning("Failed to load cli-config.yaml: %s", e)

    # Expand ${ENV_VAR} references in config values before bridging to env vars.
    from hermes_cli.config import _expand_env_vars
    defaults = _expand_env_vars(defaults)

    # Apply terminal config to environment variables (so terminal_tool picks them up)
    terminal_config = defaults.get("terminal", {})
    
    # Normalize config key: the new config system (hermes_cli/config.py) and all
    # documentation use "backend", the legacy cli-config.yaml uses "env_type".
    # Accept both, with "backend" taking precedence (it's th

... [OUTPUT TRUNCATED - 397363 chars omitted out of 447363 total] ...

 #FFD700',
            'completion-menu.meta.completion': 'bg:#1a1a2e #888888',
            'completion-menu.meta.completion.current': 'bg:#333355 #FFBF00',
            # Clarify question panel
            'clarify-border': '#CD7F32',
            'clarify-title': '#FFD700 bold',
            'clarify-question': '#FFF8DC bold',
            'clarify-choice': '#AAAAAA',
            'clarify-selected': '#FFD700 bold',
            'clarify-active-other': '#FFD700 italic',
            'clarify-countdown': '#CD7F32',
            # Sudo password panel
            'sudo-prompt': '#FF6B6B bold',
            'sudo-border': '#CD7F32',
            'sudo-title': '#FF6B6B bold',
            'sudo-text': '#FFF8DC',
            # Dangerous command approval panel
            'approval-border': '#CD7F32',
            'approval-title': '#FF8C00 bold',
            'approval-desc': '#FFF8DC bold',
            'approval-cmd': '#AAAAAA italic',
            'approval-choice': '#AAAAAA',
            'approval-selected': '#FFD700 bold',
            # Voice mode
            'voice-prompt': '#87CEEB',
            'voice-recording': '#FF4444 bold',
            'voice-processing': '#FFA500 italic',
            'voice-status': 'bg:#1a1a2e #87CEEB',
            'voice-status-recording': 'bg:#1a1a2e #FF4444 bold',
        }
        style = PTStyle.from_dict(self._build_tui_style_dict())
        
        # Create the application
        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
            **({'cursor': _STEADY_CURSOR} if _STEADY_CURSOR is not None else {}),
        )
        self._app = app  # Store reference for clarify_callback

        # ── Fix ghost status-bar lines on terminal resize ──────────────
        # When the terminal shrinks (e.g. un-maximize), the emulator reflows
        # the previously-rendered full-width rows (status bar, input rules)
        # into multiple narrower rows.  prompt_toolkit's _on_resize handler
        # only cursor_up()s by the stored layout height, missing the extra
        # rows created by reflow — leaving ghost duplicates visible.
        #
        # Fix: before the standard erase, inflate _cursor_pos.y so the
        # cursor moves up far enough to cover the reflowed ghost content.
        _original_on_resize = app._on_resize

        def _resize_clear_ghosts():
            from prompt_toolkit.data_structures import Point as _Pt
            renderer = app.renderer
            try:
                old_size = renderer._last_size
                new_size = renderer.output.get_size()
                if (
                    old_size
                    and new_size.columns < old_size.columns
                    and new_size.columns > 0
                ):
                    reflow_factor = (
                        (old_size.columns + new_size.columns - 1)
                        // new_size.columns
                    )
                    last_h = (
                        renderer._last_screen.height
                        if renderer._last_screen
                        else 0
                    )
                    extra = last_h * (reflow_factor - 1)
                    if extra > 0:
                        renderer._cursor_pos = _Pt(
                            x=renderer._cursor_pos.x,
                            y=renderer._cursor_pos.y + extra,
                        )
            except Exception:
                pass  # never break resize handling
            _original_on_resize()

        app._on_resize = _resize_clear_ghosts

        def spinner_loop():
            import time as _time

            last_idle_refresh = 0.0
            while not self._should_exit:
                if not self._app:
                    _time.sleep(0.1)
                    continue
                if self._command_running:
                    self._invalidate(min_interval=0.1)
                    _time.sleep(0.1)
                else:
                    now = _time.monotonic()
                    if now - last_idle_refresh >= 1.0:
                        last_idle_refresh = now
                        self._invalidate(min_interval=1.0)
                    _time.sleep(0.2)

        spinner_thread = threading.Thread(target=spinner_loop, daemon=True)
        spinner_thread.start()
        
        # Background thread to process inputs and run agent
        def process_loop():
            while not self._should_exit:
                try:
                    # Check for pending input with timeout
                    try:
                        user_input = self._pending_input.get(timeout=0.1)
                    except queue.Empty:
                        # Periodic config watcher — auto-reload MCP on mcp_servers change
                        if not self._agent_running:
                            self._check_config_mcp_changes()
                            # Check for background process notifications (completions
                            # and watch pattern matches) while agent is idle.
                            try:
                                from tools.process_registry import process_registry
                                if not process_registry.completion_queue.empty():
                                    evt = process_registry.completion_queue.get_nowait()
                                    # Skip if the agent already consumed this via wait/poll/log
                                    _evt_sid = evt.get("session_id", "")
                                    if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
                                        pass  # already delivered via tool result
                                    else:
                                        _synth = _format_process_notification(evt)
                                        if _synth:
                                            self._pending_input.put(_synth)
                            except Exception:
                                pass
                        continue
                    
                    if not user_input:
                        continue

                    # Unpack image payload: (text, [Path, ...]) or plain str
                    submit_images = []
                    if isinstance(user_input, tuple):
                        user_input, submit_images = user_input
                    
                    # Check for commands — but detect dragged/pasted file paths first.
                    # See _detect_file_drop() for details.
                    _file_drop = _detect_file_drop(user_input) if isinstance(user_input, str) else None
                    if _file_drop:
                        _drop_path = _file_drop["path"]
                        _remainder = _file_drop["remainder"]
                        if _file_drop["is_image"]:
                            submit_images.append(_drop_path)
                            user_input = _remainder or f"[User attached image: {_drop_path.name}]"
                            _cprint(f"  📎 Auto-attached image: {_drop_path.name}")
                        else:
                            _cprint(f"  📄 Detected file: {_drop_path.name}")
                            user_input = (
                                f"[User attached file: {_drop_path}]"
                                + (f"\n{_remainder}" if _remainder else "")
                            )

                    if not _file_drop and isinstance(user_input, str) and _looks_like_slash_command(user_input):
                        _cprint(f"\n⚙️  {user_input}")
                        if not self.process_command(user_input):
                            self._should_exit = True
                            # Schedule app exit
                            if app.is_running:
                                app.exit()
                        continue
                    
                    # Expand paste references back to full content
                    import re as _re
                    _paste_ref_re = _re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')
                    paste_refs = list(_paste_ref_re.finditer(user_input)) if isinstance(user_input, str) else []
                    if paste_refs:
                        def _expand_ref(m):
                            p = Path(m.group(1))
                            return p.read_text(encoding="utf-8") if p.exists() else m.group(0)
                        expanded = _paste_ref_re.sub(_expand_ref, user_input)
                        total_lines = expanded.count('\n') + 1
                        n_pastes = len(paste_refs)
                        _user_bar = f"[{_accent_hex()}]{'─' * 40}[/]"
                        print()
                        ChatConsole().print(_user_bar)
                        # Show any surrounding user text alongside the paste summary
                        split_parts = _paste_ref_re.split(user_input)
                        visible_user_text = " ".join(
                            split_parts[i].strip() for i in range(0, len(split_parts), 2) if split_parts[i].strip()
                        )
                        if visible_user_text:
                            ChatConsole().print(
                                f"[bold {_accent_hex()}]\u25cf[/] [bold]{_escape(visible_user_text)}[/] "
                                f"[dim]({n_pastes} pasted block{'s' if n_pastes > 1 else ''}, {total_lines} lines total)[/]"
                            )
                        else:
                            ChatConsole().print(
                                f"[bold {_accent_hex()}]\u25cf[/] [bold]{_escape(f'[Pasted text: {total_lines} lines]')}[/]"
                            )
                        user_input = expanded
                    else:
                        _user_bar = f"[{_accent_hex()}]{'─' * 40}[/]"
                        if '\n' in user_input:
                            first_line = user_input.split('\n')[0]
                            line_count = user_input.count('\n') + 1
                            print()
                            ChatConsole().print(_user_bar)
                            ChatConsole().print(
                                f"[bold {_accent_hex()}]●[/] [bold]{_escape(first_line)}[/] "
                                f"[dim](+{line_count - 1} lines)[/]"
                            )
                        else:
                            print()
                            ChatConsole().print(_user_bar)
                            ChatConsole().print(f"[bold {_accent_hex()}]●[/] [bold]{_escape(user_input)}[/]")
                    
                    # Show image attachment count
                    if submit_images:
                        n = len(submit_images)
                        _cprint(f"  {_DIM}📎 {n} image{'s' if n > 1 else ''} attached{_RST}")

                    # Regular chat - run agent
                    self._agent_running = True
                    app.invalidate()  # Refresh status line

                    try:
                        self.chat(user_input, images=submit_images or None)
                    finally:
                        self._agent_running = False
                        self._spinner_text = ""
                        self._tool_start_time = 0.0
                        self._pending_tool_info.clear()
                        self._last_scrollback_tool = ""

                        app.invalidate()  # Refresh status line

                        # Continuous voice: auto-restart recording after agent responds.
                        # Dispatch to a daemon thread so play_beep (sd.wait) and
                        # AudioRecorder.start (lock acquire) never block process_loop —
                        # otherwise queued user input would stall silently.
                        if self._voice_mode and self._voice_continuous and not self._voice_recording:
                            def _restart_recording():
                                try:
                                    if self._voice_tts:
                                        self._voice_tts_done.wait(timeout=60)
                                        time.sleep(0.3)
                                    self._voice_start_recording()
                                    app.invalidate()
                                except Exception as e:
                                    _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
                            threading.Thread(target=_restart_recording, daemon=True).start()

                        # Drain process notifications (completions + watch matches)
                        # that arrived while the agent was running.
                        try:
                            from tools.process_registry import process_registry
                            while not process_registry.completion_queue.empty():
                                evt = process_registry.completion_queue.get_nowait()
                                # Skip if the agent already consumed this via wait/poll/log
                                _evt_sid = evt.get("session_id", "")
                                if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
                                    continue  # already delivered via tool result
                                _synth = _format_process_notification(evt)
                                if _synth:
                                    self._pending_input.put(_synth)
                        except Exception:
                            pass  # Non-fatal — don't break the main loop

                except Exception as e:
                    print(f"Error: {e}")
        
        # Start processing thread
        process_thread = threading.Thread(target=process_loop, daemon=True)
        process_thread.start()
        
        # Register atexit cleanup so resources are freed even on unexpected exit
        atexit.register(_run_cleanup)
        
        # Register signal handlers for graceful shutdown on SSH disconnect / SIGTERM
        def _signal_handler(signum, frame):
            """Handle SIGHUP/SIGTERM by triggering graceful cleanup."""
            logger.debug("Received signal %s, triggering graceful shutdown", signum)
            raise KeyboardInterrupt()
        
        try:
            import signal as _signal
            _signal.signal(_signal.SIGTERM, _signal_handler)
            if hasattr(_signal, 'SIGHUP'):
                _signal.signal(_signal.SIGHUP, _signal_handler)
        except Exception:
            pass  # Signal handlers may fail in restricted environments
        
        # Install a custom asyncio exception handler that suppresses the
        # "Event loop is closed" RuntimeError from httpx transport cleanup
        # and the "0 is not registered" KeyError from broken stdin (#6393).
        # The RuntimeError fix is defense-in-depth — the primary fix is
        # neuter_async_httpx_del which disables __del__ entirely.  The
        # KeyError fix handles macOS + uv-managed Python environments where
        # fd 0 is not reliably available to the asyncio selector.
        def _suppress_closed_loop_errors(loop, context):
            exc = context.get("exception")
            if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
                return  # silently suppress
            if isinstance(exc, KeyError) and "is not registered" in str(exc):
                return  # suppress selector registration failures (#6393)
            # Fall back to default handler for everything else
            loop.default_exception_handler(context)

        # Validate stdin before launching prompt_toolkit — on macOS with
        # uv-managed Python, fd 0 can be invalid or unregisterable with the
        # asyncio selector, causing "KeyError: '0 is not registered'" (#6393).
        try:
            import os as _os
            _os.fstat(0)
        except OSError:
            print(
                "Error: stdin (fd 0) is not available.\n"
                "This can happen with certain Python installations (e.g. uv-managed cPython on macOS).\n"
                "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
            )
            _run_cleanup()
            self._print_exit_summary()
            return

        # Run the application with patch_stdout for proper output handling
        try:
            with patch_stdout():
                # Set the custom handler on prompt_toolkit's event loop
                try:
                    import asyncio as _aio
                    _loop = _aio.get_event_loop()
                    _loop.set_exception_handler(_suppress_closed_loop_errors)
                except Exception:
                    pass
                app.run()
        except (EOFError, KeyboardInterrupt, BrokenPipeError):
            pass
        except (KeyError, OSError) as _stdin_err:
            # Catch selector registration failures from broken stdin (#6393).
            # This is the fallback for cases that slip past the fstat() guard.
            if "is not registered" in str(_stdin_err) or "Bad file descriptor" in str(_stdin_err):
                print(
                    f"\nError: stdin is not usable ({_stdin_err}).\n"
                    "This can happen with certain Python installations (e.g. uv-managed cPython on macOS).\n"
                    "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
                )
            else:
                raise
        finally:
            self._should_exit = True
            # Interrupt the agent immediately so its daemon thread stops making
            # API calls and exits promptly (agent_thread is daemon, so the
            # process will exit once the main thread finishes, but interrupting
            # avoids wasted API calls and lets run_conversation clean up).
            if self.agent and getattr(self, '_agent_running', False):
                try:
                    self.agent.interrupt()
                except Exception:
                    pass
            # Flush memories before exit (only for substantial conversations)
            if self.agent and self.conversation_history:
                try:
                    self.agent.flush_memories(self.conversation_history)
                except (Exception, KeyboardInterrupt):
                    pass
            # Shut down voice recorder (release persistent audio stream)
            if hasattr(self, '_voice_recorder') and self._voice_recorder:
                try:
                    self._voice_recorder.shutdown()
                except Exception:
                    pass
                self._voice_recorder = None
            # Clean up old temp voice recordings
            try:
                from tools.voice_mode import cleanup_temp_recordings
                cleanup_temp_recordings()
            except Exception:
                pass
            # Unregister callbacks to avoid dangling references
            set_sudo_password_callback(None)
            set_approval_callback(None)
            set_secret_capture_callback(None)
            # Close session in SQLite
            if hasattr(self, '_session_db') and self._session_db and self.agent:
                try:
                    self._session_db.end_session(self.agent.session_id, "cli_close")
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not close session in DB: %s", e)
            # Plugin hook: on_session_end — safety net for interrupted exits.
            # run_conversation() already fires this per-turn on normal completion,
            # so only fire here if the agent was mid-turn (_agent_running) when
            # the exit occurred, meaning run_conversation's hook didn't fire.
            if self.agent and getattr(self, '_agent_running', False):
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_end",
                        session_id=self.agent.session_id,
                        completed=False,
                        interrupted=True,
                        model=getattr(self.agent, 'model', None),
                        platform=getattr(self.agent, 'platform', None) or "cli",
                    )
                except Exception:
                    pass
            _run_cleanup()
            self._print_exit_summary()


# ============================================================================
# Main Entry Point
# ============================================================================

def main(
    query: str = None,
    q: str = None,
    image: str = None,
    toolsets: str = None,
    skills: str | list[str] | tuple[str, ...] = None,
    model: str = None,
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
    max_turns: int = None,
    verbose: bool = False,
    quiet: bool = False,
    compact: bool = False,
    list_tools: bool = False,
    list_toolsets: bool = False,
    gateway: bool = False,
    resume: str = None,
    worktree: bool = False,
    w: bool = False,
    checkpoints: bool = False,
    pass_session_id: bool = False,
):
    """
    Hermes Agent CLI - Interactive AI Assistant
    
    Args:
        query: Single query to execute (then exit). Alias: -q
        q: Shorthand for --query
        image: Optional local image path to attach to a single query
        toolsets: Comma-separated list of toolsets to enable (e.g., "web,terminal")
        skills: Comma-separated or repeated list of skills to preload for the session
        model: Model to use (default: anthropic/claude-opus-4-20250514)
        provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
        api_key: API key for authentication
        base_url: Base URL for the API
        max_turns: Maximum tool-calling iterations (default: 60)
        verbose: Enable verbose logging
        compact: Use compact display mode
        list_tools: List available tools and exit
        list_toolsets: List available toolsets and exit
        resume: Resume a previous session by its ID (e.g., 20260225_143052_a1b2c3)
        worktree: Run in an isolated git worktree (for parallel agents). Alias: -w
        w: Shorthand for --worktree
    
    Examples:
        python cli.py                            # Start interactive mode
        python cli.py --toolsets web,terminal    # Use specific toolsets
        python cli.py --skills hermes-agent-dev,github-auth
        python cli.py -q "What is Python?"       # Single query mode
        python cli.py -q "Describe this" --image ~/storage/shared/Pictures/cat.png
        python cli.py --list-tools               # List tools and exit
        python cli.py --resume 20260225_143052_a1b2c3  # Resume session
        python cli.py -w                         # Start in isolated git worktree
        python cli.py -w -q "Fix issue #123"     # Single query in worktree
    """
    global _active_worktree

    # Signal to terminal_tool that we're in interactive mode
    # This enables interactive sudo password prompts with timeout
    os.environ["HERMES_INTERACTIVE"] = "1"
    
    # Handle gateway mode (messaging + cron)
    if gateway:
        import asyncio
        from gateway.run import start_gateway
        print("Starting Hermes Gateway (messaging platforms)...")
        asyncio.run(start_gateway())
        return

    # Skip worktree for list commands (they exit immediately)
    if not list_tools and not list_toolsets:
        # ── Git worktree isolation (#652) ──
        # Create an isolated worktree so this agent instance doesn't collide
        # with other agents working on the same repo.
        use_worktree = worktree or w or CLI_CONFIG.get("worktree", False)
        wt_info = None
        if use_worktree:
            # Prune stale worktrees from crashed/killed sessions
            _repo = _git_repo_root()
            if _repo:
                _prune_stale_worktrees(_repo)
            wt_info = _setup_worktree()
            if wt_info:
                _active_worktree = wt_info
                os.environ["TERMINAL_CWD"] = wt_info["path"]
                atexit.register(_cleanup_worktree, wt_info)
            else:
                # Worktree was explicitly requested but setup failed —
                # don't silently run without isolation.
                return
    else:
        wt_info = None
    
    # Handle query shorthand
    query = query or q
    
    # Parse toolsets - handle both string and tuple/list inputs
    # Default to hermes-cli toolset which includes cronjob management tools
    toolsets_list = None
    if toolsets:
        if isinstance(toolsets, str):
            toolsets_list = [t.strip() for t in toolsets.split(",")]
        elif isinstance(toolsets, (list, tuple)):
            # Fire may pass multiple --toolsets as a tuple
            toolsets_list = []
            for t in toolsets:
                if isinstance(t, str):
                    toolsets_list.extend([x.strip() for x in t.split(",")])
                else:
                    toolsets_list.append(str(t))
    else:
        # Use the shared resolver so MCP servers are included at runtime
        from hermes_cli.tools_config import _get_platform_tools
        toolsets_list = sorted(_get_platform_tools(CLI_CONFIG, "cli"))
    
    parsed_skills = _parse_skills_argument(skills)

    # Create CLI instance
    cli = HermesCLI(
        model=model,
        toolsets=toolsets_list,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        max_turns=max_turns,
        verbose=verbose,
        compact=compact,
        resume=resume,
        checkpoints=checkpoints,
        pass_session_id=pass_session_id,
    )

    if parsed_skills:
        skills_prompt, loaded_skills, missing_skills = build_preloaded_skills_prompt(
            parsed_skills,
            task_id=cli.session_id,
        )
        if missing_skills:
            missing_display = ", ".join(missing_skills)
            raise ValueError(f"Unknown skill(s): {missing_display}")
        if skills_prompt:
            cli.system_prompt = "\n\n".join(
                part for part in (cli.system_prompt, skills_prompt) if part
            ).strip()
            cli.preloaded_skills = loaded_skills

    # Inject worktree context into agent's system prompt
    if wt_info:
        wt_note = (
            f"\n\n[System note: You are working in an isolated git worktree at "
            f"{wt_info['path']}. Your branch is `{wt_info['branch']}`. "
            f"Changes here do not affect the main working tree or other agents. "
            f"Remember to commit and push your changes, and create a PR if appropriate. "
            f"The original repo is at {wt_info['repo_root']}.]"
        )
        cli.system_prompt = (cli.system_prompt or "") + wt_note
    
    # Handle list commands (don't init agent for these)
    if list_tools:
        cli.show_banner()
        cli.show_tools()
        sys.exit(0)
    
    if list_toolsets:
        cli.show_banner()
        cli.show_toolsets()
        sys.exit(0)
    
    # Register cleanup for single-query mode (interactive mode registers in run())
    atexit.register(_run_cleanup)
    
    # Handle single query mode
    if query or image:
        query, single_query_images = _collect_query_images(query, image)
        if quiet:
            # Quiet mode: suppress banner, spinner, tool previews.
            # Only print the final response and parseable session info.
            cli.tool_progress_mode = "off"
            if cli._ensure_runtime_credentials():
                effective_query = query
                if single_query_images:
                    effective_query = cli._preprocess_images_with_vision(
                        query,
                        single_query_images,
                        announce=False,
                    )
                turn_route = cli._resolve_turn_agent_config(effective_query)
                if turn_route["signature"] != cli._active_agent_route_signature:
                    cli.agent = None
                if cli._init_agent(
                    model_override=turn_route["model"],
                    runtime_override=turn_route["runtime"],
                    route_label=turn_route["label"],
                    request_overrides=turn_route.get("request_overrides"),
                ):
                    cli.agent.quiet_mode = True
                    cli.agent.suppress_status_output = True
                    result = cli.agent.run_conversation(
                        user_message=effective_query,
                        conversation_history=cli.conversation_history,
                    )
                    response = result.get("final_response", "") if isinstance(result, dict) else str(result)
                    if response:
                        print(response)
                    print(f"\nsession_id: {cli.session_id}")
                    
                    # Ensure proper exit code for automation wrappers
                    sys.exit(1 if isinstance(result, dict) and result.get("failed") else 0)
            
            # Exit with error code if credentials or agent init fails
            sys.exit(1)
        else:
            cli.show_banner()
            _query_label = query or ("[image attached]" if single_query_images else "")
            if _query_label:
                cli.console.print(f"[bold blue]Query:[/] {_query_label}")
            cli.chat(query, images=single_query_images or None)
            cli._print_exit_summary()
        return
    
    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    fire.Fire(main)