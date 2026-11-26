import os
import uuid

# A TensorBoard logger with a wandb/swanlab-compatible interface.
# Every `wandb.log({...})` / `lm_checkpoint(wandb=...)` call in the training
# scripts can reuse it as-is, writing loss / aux_loss / lr into TensorBoard
# event files (for `tensorboard --logdir`) without touching the main training loop.


class TensorBoardLogger:
    def __init__(self, log_dir='../runs', project='MiniMind', name=None,
                 id=None, resume=None, start_step=0):
        from torch.utils.tensorboard import SummaryWriter
        self.project = project
        self.name = name
        # One subdirectory per project: pretrain / sft runs never overwrite each other,
        # but all live under the same log_dir so a single `--logdir` shows them all.
        run_dir = os.path.join(log_dir, project)
        self._writer = SummaryWriter(run_dir)
        self.id = id or uuid.uuid4().hex[:12]
        self._step = int(start_step)

    def log(self, data, step=None):
        # Only write numeric scalars; the x axis uses an internal auto-increment counter (same default behavior as wandb).
        for key, value in data.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            try:
                self._writer.add_scalar(key, float(value), global_step=self._step)
            except (TypeError, ValueError):
                continue
        self._writer.flush()
        self._step += 1

    def get_run(self):
        # lm_checkpoint records checkpoints via `wandb.get_run().id`, so returning self is sufficient.
        return self

    @property
    def step(self):
        return self._step

    def finish(self):
        if getattr(self, '_writer', None) is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
