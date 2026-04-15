"""Tests for user-defined quick commands that bypass the agent loop."""
import subprocess
from unittest.mock import MagicMock, patch, AsyncMock
from rich.text import Text
import pytest


# ── CLI tests ──────────────────────────────────────────────────────────────

class TestCLIQuickCommands:
    """Test quick command dispatch in HermesCLI.process_command."""

    @staticmethod
    def _printed_plain(call_arg):
        if isinstance(call_arg, Text):
            return call_arg.plain
        return str(call_arg)

    def _make_cli(self, quick_commands):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.config = {"quick_commands": quick_commands}
        cli.console = MagicMock()
        cli.agent = None
        cli.conversation_history = []
        return cli

    def test_exec_command_runs_and_prints_output(self):
        cli = self._make_cli({"dn": {"type": "exec", "command": "echo daily-note"}})
        result = cli.process_command("/dn")
        assert result is True
        cli.console.print.assert_called_once()
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert printed == "daily-note"

    def test_exec_command_can_chain_post_command(self):
        cli = self._make_cli({"sw": {"type": "exec", "command": "echo switched", "post_command": "/new"}})
        cli.new_session = MagicMock()
        result = cli.process_command("/sw")
        assert result is True
        cli.new_session.assert_called_once_with()

    def test_exec_command_stderr_shown_on_no_stdout(self):
        cli = self._make_cli({"err": {"type": "exec", "command": "echo error >&2"}})
        result = cli.process_command("/err")
        assert result is True
        # stderr fallback — should print something
        cli.console.print.assert_called_once()

    def test_exec_command_no_output_shows_fallback(self):
        cli = self._make_cli({"empty": {"type": "exec", "command": "true"}})
        cli.process_command("/empty")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "no output" in args.lower()

    def test_alias_command_routes_to_target(self):
        """Alias quick commands rewrite to the target command."""
        cli = self._make_cli({"shortcut": {"type": "alias", "target": "/help"}})
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/shortcut")
            # Should recursively call process_command with /help
            spy.assert_any_call("/help")

    def test_alias_command_passes_args(self):
        """Alias quick commands forward user arguments to the target."""
        cli = self._make_cli({"sc": {"type": "alias", "target": "/context"}})
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/sc some args")
            spy.assert_any_call("/context some args")

    def test_alias_command_can_chain_post_command(self):
        cli = self._make_cli({"shortcut": {"type": "alias", "target": "/help", "post_command": "/new"}})
        cli.new_session = MagicMock()
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/shortcut")
            spy.assert_any_call("/help")
            spy.assert_any_call("/new")
        cli.new_session.assert_called_once_with()

    def test_alias_no_target_shows_error(self):
        cli = self._make_cli({"broken": {"type": "alias", "target": ""}})
        cli.process_command("/broken")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "no target defined" in args.lower()

    def test_unsupported_type_shows_error(self):
        cli = self._make_cli({"bad": {"type": "prompt", "command": "echo hi"}})
        cli.process_command("/bad")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "unsupported type" in args.lower()

    def test_missing_command_field_shows_error(self):
        cli = self._make_cli({"oops": {"type": "exec"}})
        cli.process_command("/oops")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "no command defined" in args.lower()

    def test_post_command_self_reference_shows_error(self):
        cli = self._make_cli({"oops": {"type": "exec", "command": "echo hi", "post_command": "/oops"}})
        cli.process_command("/oops")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "cannot use itself as post_command" in args.lower()

    def test_quick_command_takes_priority_over_skill_commands(self):
        """Quick commands must be checked before skill slash commands."""
        cli = self._make_cli({"mygif": {"type": "exec", "command": "echo overridden"}})
        with patch("cli._skill_commands", {"/mygif": {"name": "gif-search"}}):
            cli.process_command("/mygif")
        cli.console.print.assert_called_once()
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert printed == "overridden"

    def test_quick_command_reload_picks_up_new_config_without_restarting_cli(self):
        """A long-lived CLI should see newly-added quick commands from config.yaml."""
        cli = self._make_cli({})
        with patch("cli._QUICK_COMMANDS_CONFIG_BASELINE_MTIME", 1), \
             patch("cli._quick_commands_config_mtime", return_value=2), \
             patch("hermes_cli.config.load_config", return_value={
                 "quick_commands": {"tk1": {"type": "alias", "target": "/help"}}
             }):
            with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
                cli.process_command("/tk1")
                spy.assert_any_call("/help")

    def test_quick_command_reload_drops_removed_command_from_live_config(self):
        cli = self._make_cli({"tk1": {"type": "alias", "target": "/help"}})
        with patch("cli._QUICK_COMMANDS_CONFIG_BASELINE_MTIME", 1), \
             patch("cli._quick_commands_config_mtime", return_value=2), \
             patch("hermes_cli.config.load_config", return_value={"quick_commands": {}}):
            with patch("cli._cprint") as mock_cprint:
                cli.process_command("/tk1")
        mock_cprint.assert_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "unknown command" in printed.lower()

    def test_unknown_command_still_shows_error(self):
        cli = self._make_cli({})
        with patch("cli._cprint") as mock_cprint:
            cli.process_command("/nonexistent")
            mock_cprint.assert_called()
            printed = " ".join(str(c) for c in mock_cprint.call_args_list)
            assert "unknown command" in printed.lower()

    def test_timeout_shows_error(self):
        cli = self._make_cli({"slow": {"type": "exec", "command": "sleep 100"}})
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sleep", 30)):
            cli.process_command("/slow")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "timed out" in args.lower()


# ── Gateway tests ──────────────────────────────────────────────────────────

class TestGatewayQuickCommands:
    """Test quick command dispatch in GatewayRunner._handle_message."""

    def _make_event(self, command, args=""):
        event = MagicMock()
        event.text = f"/{command} {args}".strip()
        event.get_command.side_effect = lambda: event.text.split(maxsplit=1)[0][1:].lower() if event.text.startswith("/") else None
        event.get_command_args.side_effect = lambda: event.text.split(maxsplit=1)[1] if len(event.text.split(maxsplit=1)) > 1 else ""
        event.source = MagicMock()
        event.source.user_id = "test_user"
        event.source.user_name = "Test User"
        event.source.platform.value = "telegram"
        event.source.chat_type = "dm"
        event.source.chat_id = "123"
        return event

    @pytest.mark.asyncio
    async def test_exec_command_returns_output(self):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"limits": {"type": "exec", "command": "echo ok"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("limits")
        result = await runner._handle_message(event)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_exec_command_can_chain_post_command(self):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"limits": {"type": "exec", "command": "echo ok", "post_command": "/new"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)
        runner.hooks = MagicMock()
        runner.hooks.emit = AsyncMock()
        runner._handle_reset_command = AsyncMock(return_value="reset ok")

        event = self._make_event("limits")
        result = await runner._handle_message(event)
        assert result == "ok\n\nreset ok"
        runner._handle_reset_command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unsupported_type_returns_error(self):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"bad": {"type": "prompt", "command": "echo hi"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("bad")
        result = await runner._handle_message(event)
        assert result is not None
        assert "unsupported type" in result.lower()

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        from gateway.run import GatewayRunner
        import asyncio
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"slow": {"type": "exec", "command": "sleep 100"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("slow")
        async def _raise_timeout(awaitable, timeout):
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise asyncio.TimeoutError
        with patch("asyncio.wait_for", side_effect=_raise_timeout):
            result = await runner._handle_message(event)
        assert result is not None
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_gateway_config_object_supports_quick_commands(self):
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            quick_commands={"limits": {"type": "exec", "command": "echo ok"}}
        )
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("limits")
        result = await runner._handle_message(event)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_gateway_reload_picks_up_new_quick_command_without_restart(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("tk1")
        with patch("gateway.run._QUICK_COMMANDS_CONFIG_BASELINE_MTIME", 1), \
             patch("gateway.run._quick_commands_config_mtime", return_value=2), \
             patch("hermes_cli.config.load_config", return_value={
                 "quick_commands": {"tk1": {"type": "exec", "command": "echo live"}}
             }):
            result = await runner._handle_message(event)
        assert result == "live"

    @pytest.mark.asyncio
    async def test_gateway_reload_drops_removed_command_from_live_config(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"tk1": {"type": "exec", "command": "echo stale"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("tk1")
        with patch("gateway.run._QUICK_COMMANDS_CONFIG_BASELINE_MTIME", 1), \
             patch("gateway.run._quick_commands_config_mtime", return_value=2), \
             patch("hermes_cli.config.load_config", return_value={"quick_commands": {}}):
            result = await runner._handle_message(event)
        assert result is not None
        assert "unknown command" in result.lower()
