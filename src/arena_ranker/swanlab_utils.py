from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from arena_ranker.config import AppConfig


LOGGER = logging.getLogger("arena_ranker.swanlab")


class SwanlabTracker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = config.swanlab.enabled
        self._run: Any | None = None
        self._swanlab: Any | None = None

    def start(self) -> None:
        if not self.enabled:
            return

        try:
            import swanlab
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "已启用 SwanLab，但当前环境未安装 swanlab。请先执行 `uv sync` 安装依赖。"
            ) from exc

        init_kwargs = {
            "project": self.config.swanlab.project,
            "config": asdict(self.config),
        }
        if self.config.swanlab.experiment_name:
            init_kwargs["experiment_name"] = self.config.swanlab.experiment_name
        if self.config.swanlab.workspace:
            init_kwargs["workspace"] = self.config.swanlab.workspace
        if self.config.swanlab.mode:
            init_kwargs["mode"] = self.config.swanlab.mode

        self._swanlab = swanlab
        self._run = swanlab.init(**init_kwargs)
        LOGGER.info(
            "SwanLab 已启用: project=%s%s",
            self.config.swanlab.project,
            f", experiment_name={self.config.swanlab.experiment_name}" if self.config.swanlab.experiment_name else "",
        )

    def log(self, data: dict[str, float | int | str], step: int | None = None) -> None:
        if not self.enabled or self._swanlab is None:
            return
        if step is None:
            self._swanlab.log(data)
            return
        self._swanlab.log(data, step=step)

    def finish(self) -> None:
        if not self.enabled:
            return

        if self._run is not None and hasattr(self._run, "finish"):
            self._run.finish()
            return

        if self._swanlab is not None and hasattr(self._swanlab, "finish"):
            self._swanlab.finish()
