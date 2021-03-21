import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
from .losses import BCEWithLogitsLoss


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True, aux_weight=0.05, aux_loss=BCEWithLogitsLoss()):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self.aux_weight = aux_weight
        self.aux_loss = aux_loss

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y, i, size):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for i, (x, y) in enumerate(iterator):
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y, i, len(dataloader))

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, aux_weight=0.05, aux_loss=BCEWithLogitsLoss(),
                 grad_accumulation_steps=1,
                 use_amp=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            aux_weight=aux_weight,
            aux_loss=aux_loss,
        )
        self.optimizer = optimizer

        self.grad_accumulation_steps = grad_accumulation_steps
        self.optimizer.zero_grad()
        
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, i, size):
        accum_bin = i // self.grad_accumulation_steps
        accum_left = accum_bin * self.grad_accumulation_steps
        accum_right = min((accum_bin + 1) * self.grad_accumulation_steps, size)
        accum_n = accum_right - accum_left

        assert 1 <= accum_n <= self.grad_accumulation_steps

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if hasattr(self.model, 'classification_head') and self.model.classification_head is not None:
                prediction, label_prediction = self.model.forward(x)
                label_y = y.view(y.shape[0], y.shape[1], -1).sum(axis=2).bool().float()
                loss_label = self.aux_loss(label_prediction, label_y)
                loss = (1-self.aux_weight) * self.loss(prediction, y) + self.aux_weight * loss_label
            else:
                prediction = self.model.forward(x)
                loss = self.loss(prediction, y)
            loss = loss / accum_n
        self.scaler.scale(loss).backward()
        if i + 1 == accum_right:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        return loss * accum_n, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True, aux_weight=0.05, aux_loss=BCEWithLogitsLoss()):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            aux_weight=aux_weight,
            aux_loss=aux_loss,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, i, size):
        with torch.no_grad():
            if hasattr(self.model, 'classification_head') and self.model.classification_head is not None:
                prediction, label_prediction = self.model.forward(x)
                label_y = y.view(y.shape[0], y.shape[1], -1).sum(axis=2).bool().float()
                loss_label = self.aux_loss(label_prediction, label_y)
                loss = (1-self.aux_weight) * self.loss(prediction, y) + self.aux_weight * loss_label
            else:
                prediction = self.model.forward(x)
                loss = self.loss(prediction, y)
        return loss, prediction
