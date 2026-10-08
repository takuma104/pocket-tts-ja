"""wandb mirroring of the upstream progress log — scalars only.

The upstream trainer writes run_dir/progress.jsonl through `ProgressLog.log(event, step,
metrics, **fields)`. `make_progress_log_cls` returns a subclass that also sends the numeric
values of "train" and "valid" events to wandb. Audio, images, checkpoints and tables are
never uploaded (wandb quota). The wandb run id is stored in run_dir/wandb_id.txt so a
resumed training run and the evaluation scripts write into the same run.
"""

import os
import uuid
from pathlib import Path
from typing import Any

import wandb

from training.train_utils import ProgressLog

PROJECT = os.environ.get("WANDB_PROJECT", "pocket-tts-ja")


def run_id_for(run_dir: Path) -> str:
    path = Path(run_dir) / "wandb_id.txt"
    if path.exists():
        return path.read_text().strip()
    run_id = uuid.uuid4().hex[:12]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id + "\n")
    return run_id


def init_run(run_dir: Path, config: dict[str, Any] | None = None, job_type: str = "train") -> Any:
    """Attach to (or create) the wandb run that belongs to run_dir."""
    run_dir = Path(run_dir)
    return wandb.init(
        project=PROJECT,
        id=run_id_for(run_dir),
        name=run_dir.name,
        resume="allow",
        config=config,
        job_type=job_type,
        settings=wandb.Settings(console="off", save_code=False),
    )


def log_scalars(prefix: str, step: int, values: dict[str, Any]):
    data = {
        f"{prefix}/{k}": float(v)
        for k, v in values.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if data:
        wandb.log(data, step=step)


def make_progress_log_cls(run_dir: Path, config: dict[str, Any]) -> type[ProgressLog]:
    class WandbProgressLog(ProgressLog):
        def __init__(self, path: Path, enabled: bool = True):
            super().__init__(path, enabled)
            if enabled:
                init_run(run_dir, config)

        def log(
            self,
            event: str,
            step: int,
            metrics: dict[str, Any] | None = None,
            **fields: str | float | None,
        ):
            super().log(event, step, metrics, **fields)
            if not self.enabled or event not in ("train", "valid"):
                return
            values: dict[str, Any] = dict(metrics or {})
            values.update({k: v for k, v in fields.items() if isinstance(v, (int, float))})
            log_scalars(event, step, values)

    return WandbProgressLog
