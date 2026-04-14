"""Regression tests for the prompt_toolkit /model picker viewport.

The curses-based menus already scroll, but the inline prompt_toolkit modal used by
`/model` rendered every choice unconditionally. When the model list exceeded the
available terminal rows, lower items were never shown even though arrow-key
selection kept moving.

These tests exercise the shared viewport helper added for the prompt_toolkit
picker so the highlighted row always remains visible.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli import _model_picker_visible_window


def test_picker_window_does_not_scroll_when_all_choices_fit():
    scroll, start, end = _model_picker_visible_window(
        selected_idx=2,
        total_choices=5,
        terminal_rows=24,
        scroll_offset=0,
    )
    assert scroll == 0
    assert (start, end) == (0, 5)


def test_picker_window_scrolls_down_to_keep_lower_selection_visible():
    scroll, start, end = _model_picker_visible_window(
        selected_idx=12,
        total_choices=20,
        terminal_rows=14,
        scroll_offset=0,
    )
    assert scroll == 9
    assert (start, end) == (9, 13)
    assert start <= 12 < end


def test_picker_window_scrolls_back_up_when_selection_moves_up():
    scroll, start, end = _model_picker_visible_window(
        selected_idx=3,
        total_choices=20,
        terminal_rows=14,
        scroll_offset=9,
    )
    assert scroll == 3
    assert (start, end) == (3, 7)
    assert start <= 3 < end


def test_picker_window_keeps_cursor_visible_during_full_navigation():
    terminal_rows = 16
    total_choices = 18
    scroll_offset = 0

    for selected in range(total_choices):
        scroll_offset, start, end = _model_picker_visible_window(
            selected_idx=selected,
            total_choices=total_choices,
            terminal_rows=terminal_rows,
            scroll_offset=scroll_offset,
        )
        assert start <= selected < end, (selected, start, end)

    for selected in range(total_choices - 1, -1, -1):
        scroll_offset, start, end = _model_picker_visible_window(
            selected_idx=selected,
            total_choices=total_choices,
            terminal_rows=terminal_rows,
            scroll_offset=scroll_offset,
        )
        assert start <= selected < end, (selected, start, end)