"""
Логирование в файл и консоль.
"""

import logging

_LOGGER: logging.Logger | None = None
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger("diff_drive_nav")
        if not _LOGGER.handlers:
            _LOGGER.setLevel(logging.INFO)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            _LOGGER.addHandler(h)
    return _LOGGER


def setup_logger(
    log_file: str | None = None,
    console: bool = True,
    level: str = "INFO",
) -> logging.Logger:
    global _LOGGER
    _LOGGER = logging.getLogger("diff_drive_nav")
    _LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    _LOGGER.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        _LOGGER.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        _LOGGER.addHandler(fh)

    return _LOGGER


def log_episode(episode: int, reward: float, steps: int, success: bool, **kwargs) -> None:
    logger = _get_logger()
    parts = [f"episode={episode}", f"reward={reward:.2f}", f"steps={steps}", f"success={success}"]
    for k, v in kwargs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))


def log_message(message: str, level: str = "INFO") -> None:
    logger = _get_logger()
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, "%s", message)
