from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import _resolve_stream_delivery_flags


def test_no_edit_platform_disables_streaming_and_interim_delivery():
    flags = _resolve_stream_delivery_flags(
        adapter=SimpleNamespace(SUPPORTS_MESSAGE_EDITING=False),
        platform=Platform.WEIXIN,
        streaming_enabled=True,
        interim_messages_enabled=True,
        cursor=" ▉",
    )

    assert flags.want_stream_deltas is False
    assert flags.want_interim_messages is False
    assert flags.want_interim_consumer is False
    assert flags.effective_cursor == ""


def test_editable_platform_keeps_streaming_and_interim_delivery():
    flags = _resolve_stream_delivery_flags(
        adapter=SimpleNamespace(SUPPORTS_MESSAGE_EDITING=True),
        platform=Platform.TELEGRAM,
        streaming_enabled=True,
        interim_messages_enabled=True,
        cursor=" ▉",
    )

    assert flags.want_stream_deltas is True
    assert flags.want_interim_messages is True
    assert flags.want_interim_consumer is True
    assert flags.effective_cursor == " ▉"


def test_matrix_suppresses_cursor_but_keeps_streaming():
    flags = _resolve_stream_delivery_flags(
        adapter=SimpleNamespace(SUPPORTS_MESSAGE_EDITING=True),
        platform=Platform.MATRIX,
        streaming_enabled=True,
        interim_messages_enabled=False,
        cursor=" ▉",
    )

    assert flags.want_stream_deltas is True
    assert flags.want_interim_messages is False
    assert flags.want_interim_consumer is False
    assert flags.effective_cursor == ""