"""Tests for _stream_delta's handling of <think> tags in prose vs real reasoning blocks."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest


def _make_cli_stub():
    """Create a minimal HermesCLI-like object with stream state."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli._stream_buf = ""
    cli._stream_started = False
    cli._stream_box_opened = False
    cli._stream_prefilt = ""
    cli._in_reasoning_block = False
    cli._reasoning_stream_started = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_preview_buf = ""
    cli._deferred_content = ""
    cli._stream_text_ansi = ""
    cli._stream_needs_break = False
    cli._emitted = []

    # Mock _emit_stream_text to capture output
    def mock_emit(text):
        cli._emitted.append(text)
    cli._emit_stream_text = mock_emit

    # Mock _stream_reasoning_delta
    cli._reasoning_emitted = []
    def mock_reasoning(text):
        cli._reasoning_emitted.append(text)
    cli._stream_reasoning_delta = mock_reasoning

    return cli


def _make_cli_display_stub():
    """Create a stub that uses the real _emit_stream_text / _flush_stream methods."""
    from cli import HermesCLI

    cli = _make_cli_stub()
    cli._emit_stream_text = HermesCLI._emit_stream_text.__get__(cli, HermesCLI)
    cli._flush_stream = HermesCLI._flush_stream.__get__(cli, HermesCLI)
    cli._close_reasoning_box = lambda: None
    return cli


class TestThinkTagInProse:
    """<think> mentioned in prose should NOT trigger reasoning suppression."""

    def test_think_tag_mid_sentence(self):
        """'(/think not producing <think> tags)' should pass through."""
        cli = _make_cli_stub()
        tokens = [
            "  1. Fix reasoning mode in eval ",
            "(/think not producing ",
            "<think>",
            " tags — ~2% gap)",
            "\n  2. Launch production",
        ]
        for t in tokens:
            cli._stream_delta(t)
        assert not cli._in_reasoning_block, "<think> in prose should not enter reasoning block"
        full = "".join(cli._emitted)
        assert "<think>" in full, "The literal <think> tag should be in the emitted text"
        assert "Launch production" in full

    def test_think_tag_after_text_on_same_line(self):
        """'some text <think>' should NOT trigger reasoning."""
        cli = _make_cli_stub()
        cli._stream_delta("Here is the <think> tag explanation")
        assert not cli._in_reasoning_block
        full = "".join(cli._emitted)
        assert "<think>" in full

    def test_think_tag_in_backticks(self):
        """'`<think>`' should NOT trigger reasoning."""
        cli = _make_cli_stub()
        cli._stream_delta("Use the `<think>` tag for reasoning")
        assert not cli._in_reasoning_block


class TestRealReasoningBlock:
    """Real <think> tags at block boundaries should still be caught."""

    def test_think_at_start_of_stream(self):
        """'<think>reasoning</think>answer' should suppress reasoning."""
        cli = _make_cli_stub()
        cli._stream_delta("<think>")
        assert cli._in_reasoning_block
        cli._stream_delta("I need to analyze this")
        cli._stream_delta("</think>")
        assert not cli._in_reasoning_block
        cli._stream_delta("Here is my answer")
        full = "".join(cli._emitted)
        assert "Here is my answer" in full
        assert "I need to analyze" not in full  # reasoning was suppressed

    def test_think_after_newline(self):
        """'text\\n<think>' should trigger reasoning block."""
        cli = _make_cli_stub()
        cli._stream_delta("Some preamble\n<think>")
        assert cli._in_reasoning_block
        full = "".join(cli._emitted)
        assert "Some preamble" in full

    def test_think_after_newline_with_whitespace(self):
        """'text\\n  <think>' should trigger reasoning block."""
        cli = _make_cli_stub()
        cli._stream_delta("Some preamble\n  <think>")
        assert cli._in_reasoning_block

    def test_think_with_only_whitespace_before(self):
        """'   <think>' (whitespace only prefix) should trigger."""
        cli = _make_cli_stub()
        cli._stream_delta("   <think>")
        assert cli._in_reasoning_block


class TestInternalProtocolSuppression:
    def test_malformed_tool_protocol_line_is_suppressed(self):
        cli = _make_cli_display_stub()
        printed = []

        from unittest.mock import patch
        import shutil

        with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint", side_effect=lambda text: printed.append(text)):
                cli._stream_delta(
                    "Working on it...\n"
                    "to=multi_tool_use.parallel recipient_name=functions.terminal "
                    'parameters={"command":"pwd"} Need proper JSON.\n'
                    "Done.\n"
                )
                cli._flush_stream()

        full = "".join(line for line in printed if "╭" not in line and "╯" not in line)
        assert "Need proper JSON" not in full
        assert "recipient_name=functions.terminal" not in full
        assert "Working on it..." in full
        assert "Done." in full

    def test_raw_tool_call_block_is_suppressed(self):
        cli = _make_cli_display_stub()
        printed = []

        from unittest.mock import patch
        import shutil

        with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint", side_effect=lambda text: printed.append(text)):
                cli._stream_delta('Before\n<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>\nAfter\n')
                cli._flush_stream()

        full = "".join(line for line in printed if "╭" not in line and "╯" not in line)
        assert "<tool_call>" not in full
        assert "Before" in full
        assert "After" in full

    def test_multiline_tool_call_block_is_suppressed(self):
        cli = _make_cli_display_stub()
        printed = []

        from unittest.mock import patch
        import shutil

        with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint", side_effect=lambda text: printed.append(text)):
                cli._stream_delta("Before\n<tool_call>\n")
                cli._stream_delta('{"name":"terminal","arguments":{"command":"pwd"}}\n')
                cli._stream_delta("</tool_call>\nAfter\n")
                cli._flush_stream()

        full = "".join(line for line in printed if "╭" not in line and "╯" not in line)
        assert "<tool_call>" not in full
        assert '"name":"terminal"' not in full
        assert "Before" in full
        assert "After" in full

    def test_plain_json_status_is_preserved(self):
        cli = _make_cli_display_stub()
        printed = []

        from unittest.mock import patch
        import shutil

        with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint", side_effect=lambda text: printed.append(text)):
                cli._stream_delta('{"summary":"ready","status":"ok"}\n')
                cli._flush_stream()

        full = "".join(line for line in printed if "╭" not in line and "╯" not in line)
        assert '{"summary":"ready","status":"ok"}' in full

    def test_blank_line_in_normal_stream_is_preserved(self):
        cli = _make_cli_display_stub()
        printed = []

        from unittest.mock import patch
        import shutil

        with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint", side_effect=lambda text: printed.append(text)):
                cli._stream_delta("Line 1\n\nLine 2\n")
                cli._flush_stream()

        body = [line for line in printed if "╭" not in line and "╯" not in line]
        assert body.count("") == 1
        assert any("Line 1" in line for line in body)
        assert any("Line 2" in line for line in body)


class TestFlushRecovery:
    """_flush_stream should recover content from false-positive reasoning blocks."""

    def test_flush_recovers_buffered_content(self):
        """If somehow in reasoning block at flush, content is recovered."""
        cli = _make_cli_stub()
        # Manually set up a false-positive state
        cli._in_reasoning_block = True
        cli._stream_prefilt = " tags — ~2% gap)\n  2. Launch production"
        cli._stream_box_opened = True

        # Mock _close_reasoning_box and box closing
        cli._close_reasoning_box = lambda: None

        # Call flush
        from unittest.mock import patch
        import shutil
        with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint"):
                cli._flush_stream()

        assert not cli._in_reasoning_block
        full = "".join(cli._emitted)
        assert "Launch production" in full
