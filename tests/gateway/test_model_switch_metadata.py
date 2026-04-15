"""Regression tests for gateway transcript model metadata after /model switches."""


def test_resolve_transcript_session_model_prefers_runtime_model(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4-2026-03-05")

    assert (
        gateway_run._resolve_transcript_session_model({"model": "gpt-5-2025-08-07"})
        == "gpt-5-2025-08-07"
    )


def test_resolve_transcript_session_model_falls_back_to_config_when_runtime_missing(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4-2026-03-05")

    assert gateway_run._resolve_transcript_session_model({}) == "gpt-5.4-2026-03-05"
    assert gateway_run._resolve_transcript_session_model(None) == "gpt-5.4-2026-03-05"
