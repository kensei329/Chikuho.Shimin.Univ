"""ログ。各ステップの開始・終了・所要秒数を必ず出す（SPEC 11章）。"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

LOGGER_NAME = "radio_cutter"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """標準エラーに出す。ffmpeg の出力と混ざらないよう stderr 固定。"""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)
    return logger


def fmt_duration(seconds: float) -> str:
    """所要時間を人が読める形に。"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}分{sec:.0f}秒"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}時間{minutes}分{sec:.0f}秒"


@contextmanager
def step_timer(step_no: int | str, name: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """ステップの開始・終了・所要秒数をログに出す。

        with step_timer(3, "アンカー検出") as t:
            ...
        t["elapsed"]  # 所要秒数
    """
    log = logger or get_logger()
    info: dict = {"step": step_no, "name": name, "elapsed": 0.0}
    log.info("▶ Step %s %s 開始", step_no, name)
    started = time.perf_counter()
    try:
        yield info
    except BaseException:
        info["elapsed"] = time.perf_counter() - started
        log.error("✖ Step %s %s 失敗（%s）", step_no, name, fmt_duration(info["elapsed"]))
        raise
    else:
        info["elapsed"] = time.perf_counter() - started
        log.info("✔ Step %s %s 完了（%s）", step_no, name, fmt_duration(info["elapsed"]))
