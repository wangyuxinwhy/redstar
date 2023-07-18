import numpy as np

from redstar.types import Records


class BaseMetric:
    def __call__(self, records: Records) -> dict[str, float]:
        raise NotImplementedError


class Accuracy(BaseMetric):
    def __call__(self, records: Records) -> dict[str, float]:
        preds = np.array([record['pred'] for record in records])
        targets = np.array([record['target'] for record in records])
        acc = (preds == targets).mean()
        return {'accuracy': round(float(acc), 5)}
