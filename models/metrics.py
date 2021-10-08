import torch

import os.path as osp

EPSILON = 1e-8
ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
LABEL2ID = None

# 算精确度就在这里, TODO: 加个长度的assert.
def categorical_accuracy(y_pred, y, y_unseen):
    """Calculates categorical accuracy.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [size, num_categories]
        y: Ground truth categories. Must have shape [size,]
    """
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]

def categorical_accuracy_per_class(y_pred, y, y_unseen, base_class_num=64):
    y2y_unseen = dict(zip(y.tolist(), y_unseen.tolist()))

    acc = {}
    for l in y.unique():
        acc[str(y2y_unseen[l.item()])] = ((y_pred.argmax(dim=-1) == y) * (l == y)).sum().item() / (l == y).sum().item()
    # acc[base_class_num] = ((y_pred.argmax(dim=-1) == y) * (y >= base_class_num)).sum().item() / (y >= base_class_num).sum().item()
    acc['seen'] = ((y_pred.argmax(dim=-1) == y) * (y < base_class_num)).sum().item() / (y < base_class_num).sum().item()
    acc['unseen'] = ((y_pred.argmax(dim=-1) == y) * (y >= base_class_num)).sum().item() / (y >= base_class_num).sum().item()
    acc['harmonic_mean'] = 2 * (acc['seen'] * acc['unseen']) / (acc['seen'] + acc['unseen'] + 1e-12)
    return acc

def categorical_accuracy_onehot(y_pred, y, y_unseen):
    return torch.eq(y_pred.argmax(dim=-1), y.argmax(dim=-1)).sum().item() / y_pred.shape[0]

def k_nearest_neighbour_accuracy(y_pred, y):
    """Calculates k-NN accuracy.

    # Arguments:
        y_pred: Prediction categories [size, categories]
        y: Ground truth categories. Must have shape [size,]
    """
    # 取最多的类别:
    y_pred, _ = torch.mode(y_pred, dim=1)
    return torch.eq(y_pred, y).sum().item() / y_pred.shape[0]


NAMED_METRICS = {
    'categorical accuracy': categorical_accuracy,
    'categorical accuracy one-hot': categorical_accuracy_onehot,
    'categorical accuracy per class': categorical_accuracy_per_class
    }

# 许多计算距离都是在这儿. 注意matching_fn(distance类型)参数.
def pairwise_distances(
    x: torch.Tensor,
    y: torch.Tensor,
    matching_fn: str,
    temperature: float = 1,
    has_meta_batch_size = True
    ) -> torch.Tensor:
    # [meta_batch_size, size, feat_dim]
    if has_meta_batch_size:
        n_x = x.shape[1]
        n_y = y.shape[1]
        meta_batch_size = x.shape[0]
    else:
        n_x = x.shape[0]
        n_y = y.shape[0]

    if matching_fn == 'l2' and has_meta_batch_size:
        distances = (
            x.unsqueeze(2).expand(meta_batch_size, n_x, n_y, *x.shape[2:]) -
            y.unsqueeze(1).expand(meta_batch_size, n_x, n_y, *y.shape[2:])
        ).pow(2).sum(dim=-1) / temperature

        return distances

    elif matching_fn == 'l2' and not has_meta_batch_size:
        distances = (
            x.unsqueeze(1).expand(n_x, n_y, *x.shape[1:]) -
            y.unsqueeze(0).expand(n_x, n_y, *x.shape[1:])
        ).pow(2).sum(dim=-1) / temperature

        return distances

    elif matching_fn == 'cosine' and has_meta_batch_size:
        normalised_x = x / (x.pow(2).sum(dim=-1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=-1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(2).expand(meta_batch_size, n_x, n_y, *normalised_x.shape[2:])
        expanded_y = normalised_y.unsqueeze(1).expand(meta_batch_size, n_x, n_y, *normalised_y.shape[2:])

        cosine_similarities = (expanded_x * expanded_y).sum(dim=-1)
        return 1 - cosine_similarities

    elif matching_fn == 'cosine' and not has_meta_batch_size:
        normalised_x = x / (x.pow(2).sum(dim=-1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=-1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=-1)
        return 1 - cosine_similarities

    elif matching_fn == 'dot' and not has_meta_batch_size:
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)

    else:
        raise ValueError('Unsupported similarity function')

def metrics_handle(model, monitor):
    def batch_metrics(logits, y, y_unseen=None):
        model.eval()
        with torch.no_grad():
            # # 迭代更新每一个度量.
            # for m in metrics:
            #     logs[m] = NAMED_METRICS[m](logits, y)
            return NAMED_METRICS[monitor](logits, y, y_unseen)

    return batch_metrics
