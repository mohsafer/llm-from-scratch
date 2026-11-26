import os
import uuid



class TensorBoardLogger:
    def __init__(self, log_dir='../runs', project='MiniMind', name=None,
                 id=None, resume=None, start_step=0):
        from torch.utils.tensorboard import SummaryWriter
        self.project = project
        self.name = name
        run_dir = os.path.join(log_dir, project)
        self._writer = SummaryWriter(run_dir)
        self.id = id or uuid.uuid4().hex[:12]
        self._step = int(start_step)

    def log(self, data, step=None):
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
        return self

    @property
    def step(self):
        return self._step

    def finish(self):
        if getattr(self, '_writer', None) is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
