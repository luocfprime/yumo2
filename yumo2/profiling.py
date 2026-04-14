# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

import os
import time
import tracemalloc
from contextlib import ContextDecorator
from functools import wraps

import structlog

logger = structlog.get_logger(__name__)


def _profile_enabled(mode: str) -> bool:
    val = os.getenv("YUMO2_PROFILE", "").lower()
    return val in (mode, "all")


class time_profiler(ContextDecorator):
    """Wall-clock timer, usable as a context manager or decorator.

    Only active when ``YUMO2_PROFILE=time|all``.  Logs a single
    ``profile_elapsed`` event on exit.

    Example::

        with time_profiler("my_stage"):
            ...

        @time_profiler("my_func")
        def my_func(): ...
    """

    def __init__(self, name=None):
        self.name = name
        self._enabled = False

    def __enter__(self):
        self._enabled = _profile_enabled("time")
        if self._enabled:
            self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._enabled:
            elapsed = time.perf_counter() - self._start
            logger.debug("profile_elapsed", name=self.name, elapsed_seconds=elapsed)

    def __call__(self, func):
        prof_name = self.name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with time_profiler(prof_name):
                return func(*args, **kwargs)

        return wrapper


class mem_profiler(ContextDecorator):
    """Memory profiler, usable as a context manager or decorator.

    Only active when ``YUMO2_PROFILE=memory|all``.  Uses :mod:`tracemalloc` to
    capture peak RSS and the top-*N* allocation sites, logging them as
    ``mem_profile`` and ``mem_profile_top`` events respectively.

    Example::

        with mem_profiler("my_stage", top_n=5):
            big_array = np.zeros((10000, 10000))
    """

    def __init__(self, name=None, top_n: int = 10):
        self.name = name
        self.top_n = top_n
        self._enabled = False

    def __enter__(self):
        self._enabled = _profile_enabled("memory")
        if self._enabled:
            self._was_tracing = tracemalloc.is_tracing()
            if not self._was_tracing:
                tracemalloc.start()
            self._snapshot1 = tracemalloc.take_snapshot()
        return self

    def __exit__(self, *exc):
        if not self._enabled:
            return
        try:
            snapshot2 = tracemalloc.take_snapshot()
            current, peak = tracemalloc.get_traced_memory()
            logger.debug(
                "mem_profile",
                name=self.name,
                mem_current_mb=round(current / 1024 / 1024, 3),
                mem_peak_mb=round(peak / 1024 / 1024, 3),
            )
            stats = snapshot2.compare_to(self._snapshot1, "lineno")
            top = [s for s in stats if s.size_diff > 0][: self.top_n]
            for rank, stat in enumerate(top, start=1):
                frame = stat.traceback[0]
                logger.debug(
                    "mem_profile_top",
                    name=self.name,
                    rank=rank,
                    file=frame.filename,
                    lineno=frame.lineno,
                    size_diff_kb=round(stat.size_diff / 1024, 2),
                )
        finally:
            if not self._was_tracing:
                tracemalloc.stop()

    def __call__(self, func):
        prof_name = self.name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with mem_profiler(prof_name, self.top_n):
                return func(*args, **kwargs)

        return wrapper


class freq_time_profiler:
    """Decorator that accumulates timings over a rolling time window and logs stats periodically.

    Only active when YUMO2_PROFILE=time|all. Designed for high-frequency functions
    (e.g. render callbacks) where per-call logging would be too noisy.
    """

    def __init__(self, name: str | None = None, window_seconds: float = 5.0):
        self.name = name
        self.window_seconds = window_seconds
        self._samples: list[float] = []
        self._window_start: float | None = None

    def __call__(self, func):
        prof_name = self.name or func.__name__
        self.name = prof_name

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _profile_enabled("time"):
                return func(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self._record(time.perf_counter() - t0)

        return wrapper

    def _record(self, elapsed: float) -> None:
        now = time.perf_counter()
        if self._window_start is None:
            self._window_start = now
        self._samples.append(elapsed)
        if now - self._window_start >= self.window_seconds:
            self._flush(now)

    def _flush(self, now: float) -> None:
        import statistics

        n = len(self._samples)
        mean = statistics.mean(self._samples)
        std = statistics.stdev(self._samples) if n >= 2 else 0.0
        logger.debug(
            "profile_freq",
            name=self.name,
            count=n,
            mean_seconds=round(mean, 6),
            std_seconds=round(std, 6),
            min_seconds=round(min(self._samples), 6),
            max_seconds=round(max(self._samples), 6),
            freq_hz=round(n / self.window_seconds, 2) if self.window_seconds > 0 else None,
        )
        self._samples = []
        self._window_start = now
