from sklearn.metrics import accuracy_score

from redstar.types import Records


class BaseMetric:
    def __call__(self, records: Records) -> dict[str, float]:
        raise NotImplementedError


class Accuracy(BaseMetric):
    def __call__(self, records: Records) -> dict[str, float]:
        preds = [record['pred'] for record in records]
        targets = [record['target'] for record in records]
        acc = accuracy_score(targets, preds)
        return {'accuracy': round(float(acc), 5)}
