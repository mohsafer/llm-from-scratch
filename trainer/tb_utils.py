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

    def add_graph(self, model, input, tag='minimind'):
        import torch

        class _G(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m.forward(x)
                return out.logits if hasattr(out, 'logits') else out

        saved = [(m, m.flash) for m in model.modules() if hasattr(m, 'flash')]
        for m, _ in saved:
            m.flash = False
        try:
            self._writer.add_graph(_G(model), input, verbose=False)
        finally:
            for m, v in saved:
                m.flash = v
        self._writer.flush()

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
