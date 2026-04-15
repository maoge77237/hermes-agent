from unittest.mock import MagicMock, patch

from cli import HermesCLI


def test_model_picker_visible_choices_returns_full_list_when_it_fits():
    choices = ["provider-a", "provider-b", "Cancel"]

    visible, offset, has_above, has_below = HermesCLI._model_picker_visible_choices(
        choices,
        selected=1,
        term_lines=24,
    )

    assert visible == choices
    assert offset == 0
    assert has_above is False
    assert has_below is False


def test_model_picker_visible_choices_scrolls_selected_row_into_view():
    choices = [f"model-{i}" for i in range(1, 21)]

    visible, offset, has_above, has_below = HermesCLI._model_picker_visible_choices(
        choices,
        selected=11,
        term_lines=24,
    )

    assert visible == choices[6:16]
    assert offset == 6
    assert has_above is True
    assert has_below is True
    assert choices[11] in visible


def test_model_picker_visible_choices_clamps_to_last_page_near_end():
    choices = [f"model-{i}" for i in range(1, 21)]

    visible, offset, has_above, has_below = HermesCLI._model_picker_visible_choices(
        choices,
        selected=len(choices) - 1,
        term_lines=24,
    )

    assert visible == choices[-10:]
    assert offset == 10
    assert has_above is True
    assert has_below is False


def test_get_tui_terminal_lines_prefers_prompt_toolkit_app_height():
    mock_app = MagicMock()
    mock_app.output.get_size.return_value = MagicMock(rows=18)

    with patch("prompt_toolkit.application.get_app", return_value=mock_app), \
         patch("shutil.get_terminal_size") as mock_shutil:
        lines = HermesCLI._get_tui_terminal_lines()

    assert lines == 18
    mock_shutil.assert_not_called()


def test_panel_box_width_prefers_prompt_toolkit_width_over_shutil():
    mock_app = MagicMock()
    mock_app.output.get_size.return_value = MagicMock(columns=40)

    with patch("prompt_toolkit.application.get_app", return_value=mock_app), \
         patch("shutil.get_terminal_size") as mock_shutil:
        width = HermesCLI._panel_box_width(
            "⚙ Model Picker — Tokenx24.com",
            ["gpt-5-2025-08-07", "gpt-5.4-2026-03-05", "gpt-5.4-mini"],
            min_width=46,
            max_width=84,
        )

    assert width == 36
    mock_shutil.assert_not_called()
