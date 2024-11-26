import torch
import torchmetrics as tm

from utils.tools import to_numpy


def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return torch.sqrt((MSE(pred, true)))


def MAPE(pred, true):
    return torch.mean(torch.abs((pred - true) / true))


def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / true))


def metric_torch(pred, true):
    mae = MAE(pred, true).item()
    mse = MSE(pred, true).item()
    rmse = RMSE(pred, true).item()
    mape = MAPE(pred, true).item()
    mspe = MSPE(pred, true).item()

    return mae, mse, rmse, mape, mspe


class MeanAE(tm.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, num_outputs=1, reduction='mean', **kwargs) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.num_outputs = num_outputs if reduction == 'none' else 1

        self.add_state("sum_abs_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        if self.reduction == 'mean':
            sum_abs_error = torch.sum(torch.abs(preds - target))
            num_obs = target.numel()
        elif self.reduction == 'none':
            sum_abs_error = torch.sum(torch.abs(preds - target), dim=0)
            num_obs = target.shape[0]
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

        self.sum_abs_error += sum_abs_error
        self.total += num_obs

    def compute(self):
        """Compute mean absolute error over state."""
        value = self.sum_abs_error / self.total
        return value.item() if self.num_outputs == 1 else value.detach().cpu()


class MeanSE(tm.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, squared=True, num_outputs=1, reduction='mean', **kwargs) -> None:
        super().__init__(**kwargs)
        self.squared = squared
        self.reduction = reduction
        self.num_outputs = num_outputs if reduction == 'none' else 1

        self.add_state("sum_squared_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        if self.reduction == 'mean':
            sum_squared_error = torch.sum((preds - target) ** 2)
            num_obs = target.numel()
        elif self.reduction == 'none':
            sum_squared_error = torch.sum((preds - target) ** 2, dim=0)
            num_obs = target.shape[0]
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self):
        """Compute mean squared error over state."""
        value = self.sum_squared_error / self.total if self.squared else torch.sqrt(self.sum_squared_error / self.total)
        return value.item() if self.num_outputs == 1 else value.detach().cpu()


class MAbsPE(tm.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, num_outputs=1, reduction='mean', **kwargs) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.num_outputs = num_outputs if reduction == 'none' else 1

        self.add_state("sum_abs_per_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        abs_per_error = torch.abs((preds - target) / target)
        if self.reduction == 'mean':
            sum_abs_per_error = torch.sum(abs_per_error)
            num_obs = target.numel()
        elif self.reduction == 'none':
            sum_abs_per_error = torch.sum(abs_per_error, dim=0)
            num_obs = target.shape[0]
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self):
        """Compute mean absolute percentage error over state."""
        value = self.sum_abs_per_error / self.total
        return value.item() if self.num_outputs == 1 else value.detach().cpu()


class MSqrPE(tm.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, num_outputs=1, reduction='mean', **kwargs) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.num_outputs = num_outputs if reduction == 'none' else 1

        self.add_state("sum_sqr_per_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        if self.reduction == 'mean':
            sum_sqr_per_error = torch.sum(torch.square((preds - target) / target))
            num_obs = target.numel()
        elif self.reduction == 'none':
            sum_sqr_per_error = torch.sum(torch.square((preds - target) / target), dim=0)
            num_obs = target.shape[0]
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

        self.sum_sqr_per_error += sum_sqr_per_error
        self.total += num_obs

    def compute(self):
        """Compute mean square percentage error over state."""
        value = self.sum_sqr_per_error / self.total
        return value.item() if self.num_outputs == 1 else value.detach().cpu()


class MeanRE(tm.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, num_outputs=1, reduction='mean', epsilon=1e-6, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.num_outputs = num_outputs if reduction == 'none' else 1
        self.epsilon = epsilon

        self.add_state("sum_rel_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        rel_error = torch.abs(preds - target) / torch.abs(target)
        
        if self.reduction == 'mean':
            sum_rel_error = torch.sum(rel_error)
            num_obs = target.numel()
        elif self.reduction == 'none':
            sum_rel_error = torch.sum(rel_error, dim=0)
            num_obs = target.shape[0]
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

        self.sum_rel_error += sum_rel_error
        self.total += num_obs

    def compute(self):
        """Compute mean relative error over state."""
        value = self.sum_rel_error / self.total
        return value.item() if self.num_outputs == 1 else value.detach().cpu()


def create_metric_collector(device='cpu', num_outputs=1, reduction='mean'):
    collector = tm.MetricCollection({
        "mae": MeanAE(num_outputs=num_outputs, reduction=reduction),
        "mse": MeanSE(num_outputs=num_outputs, reduction=reduction),
        "rmse": MeanSE(squared=False, num_outputs=num_outputs, reduction=reduction),
        "mape": MAbsPE(num_outputs=num_outputs, reduction=reduction),
        "mspe": MSqrPE(num_outputs=num_outputs, reduction=reduction),
        "mre": MeanRE(num_outputs=num_outputs, reduction=reduction)
    }).to(device)
    collector.reset()
    return collector