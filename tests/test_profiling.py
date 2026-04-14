# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

# tests/test_profiling.py
from __future__ import annotations

import pytest

import yumo2.profiling as mod
from yumo2.profiling import _profile_enabled, freq_time_profiler, mem_profiler, time_profiler


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def debug(self, event: str, **fields) -> None:
        self.events.append({"event": event, **fields})


@pytest.mark.parametrize(
    ("env_value", "time_enabled", "memory_enabled"),
    [
        (None, False, False),
        ("time", True, False),
        ("memory", False, True),
        ("all", True, True),
    ],
)
def test_profile_enabled_modes(monkeypatch, env_value, time_enabled, memory_enabled):
    if env_value is None:
        monkeypatch.delenv("YUMO2_PROFILE", raising=False)
    else:
        monkeypatch.setenv("YUMO2_PROFILE", env_value)

    assert _profile_enabled("time") is time_enabled
    assert _profile_enabled("memory") is memory_enabled


def test_time_profiler_logs_when_enabled(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "time")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    with time_profiler("my_stage"):
        pass

    assert len(fake_logger.events) == 1
    assert fake_logger.events[0]["event"] == "profile_elapsed"
    assert fake_logger.events[0]["name"] == "my_stage"
    assert isinstance(fake_logger.events[0]["elapsed_seconds"], float)
    assert fake_logger.events[0]["elapsed_seconds"] >= 0.0


def test_time_profiler_no_log_when_disabled(monkeypatch):
    monkeypatch.delenv("YUMO2_PROFILE", raising=False)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    with time_profiler("my_stage"):
        pass

    assert fake_logger.events == []


def test_time_profiler_as_decorator(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "time")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    @time_profiler("decorated")
    def my_func(x):
        return x * 2

    result = my_func(3)
    assert result == 6
    assert len(fake_logger.events) == 1
    assert fake_logger.events[0]["name"] == "decorated"


def test_time_profiler_decorator_uses_func_name_when_no_name(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "time")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    @time_profiler()
    def compute_something():
        return 42

    compute_something()
    assert fake_logger.events[0]["name"] == "compute_something"


def test_mem_profiler_no_log_when_disabled(monkeypatch):
    monkeypatch.delenv("YUMO2_PROFILE", raising=False)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    with mem_profiler("my_stage"):
        _ = [0] * 1000

    assert fake_logger.events == []


def test_mem_profiler_logs_summary_when_enabled(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "memory")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    with mem_profiler("my_stage"):
        _ = [0] * 100_000

    summary_events = [e for e in fake_logger.events if e["event"] == "mem_profile"]
    assert len(summary_events) == 1
    ev = summary_events[0]
    assert ev["name"] == "my_stage"
    assert isinstance(ev["mem_current_mb"], float)
    assert isinstance(ev["mem_peak_mb"], float)
    assert ev["mem_peak_mb"] > 0.0
    assert ev["mem_peak_mb"] >= ev["mem_current_mb"]


def test_mem_profiler_logs_top_n_diff_when_enabled(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "memory")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    with mem_profiler("my_stage", top_n=3):
        _ = [0] * 100_000

    top_events = [e for e in fake_logger.events if e["event"] == "mem_profile_top"]
    assert 1 <= len(top_events) <= 3
    for ev in top_events:
        assert "rank" in ev
        assert "file" in ev
        assert "lineno" in ev
        assert "size_diff_kb" in ev
        assert ev["size_diff_kb"] > 0


def test_mem_profiler_as_decorator(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "memory")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    @mem_profiler("decorated_mem")
    def allocate():
        return [0] * 50_000

    result = allocate()
    assert len(result) == 50_000
    assert any(e["event"] == "mem_profile" and e["name"] == "decorated_mem" for e in fake_logger.events)


def test_freq_time_profiler_no_log_before_window(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "time")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    @freq_time_profiler("tick", window_seconds=60.0)
    def tick():
        pass

    for _ in range(10):
        tick()

    assert not any(e["event"] == "profile_freq" for e in fake_logger.events)


def test_freq_time_profiler_logs_after_window(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "time")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    profiler = freq_time_profiler("tick", window_seconds=0.0)

    @profiler
    def tick():
        pass

    tick()
    tick()

    events = [e for e in fake_logger.events if e["event"] == "profile_freq"]
    assert len(events) >= 1
    ev = events[0]
    assert ev["name"] == "tick"
    assert ev["count"] >= 1
    assert isinstance(ev["mean_seconds"], float)
    assert isinstance(ev["std_seconds"], float)
    assert isinstance(ev["min_seconds"], float)
    assert isinstance(ev["max_seconds"], float)
    assert ev["freq_hz"] is None or isinstance(ev["freq_hz"], float)
    assert ev["min_seconds"] <= ev["mean_seconds"] <= ev["max_seconds"]


def test_freq_time_profiler_no_log_when_disabled(monkeypatch):
    monkeypatch.delenv("YUMO2_PROFILE", raising=False)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    profiler = freq_time_profiler("tick", window_seconds=0.0)

    @profiler
    def tick():
        pass

    tick()
    tick()

    assert not any(e["event"] == "profile_freq" for e in fake_logger.events)


def test_freq_time_profiler_uses_func_name_when_no_name(monkeypatch):
    monkeypatch.setenv("YUMO2_PROFILE", "time")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod, "logger", fake_logger)

    profiler = freq_time_profiler(window_seconds=0.0)

    @profiler
    def render_frame():
        pass

    render_frame()
    render_frame()

    events = [e for e in fake_logger.events if e["event"] == "profile_freq"]
    assert any(e["name"] == "render_frame" for e in events)
