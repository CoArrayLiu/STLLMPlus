import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch


def load_graph_data(filename):
    path = Path(filename)
    if path.suffix == ".npy":
        return np.load(path)
    return load_pickle(path)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as handle:
            return pickle.load(handle)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as handle:
            return pickle.load(handle, encoding="latin1")


class DataLoader:
    """Compatibility loader for the repository's pre-windowed NPZ datasets."""

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
        self.batch_size = batch_size
        if pad_with_last_sample:
            padding = (batch_size - len(xs) % batch_size) % batch_size
            if padding:
                xs = np.concatenate([xs, np.repeat(xs[-1:], padding, axis=0)])
                ys = np.concatenate([ys, np.repeat(ys[-1:], padding, axis=0)])
        self.xs = xs
        self.ys = ys
        self.size = len(xs)
        self.num_batch = math.ceil(self.size / batch_size)

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]

    def get_iterator(self):
        for batch_index in range(self.num_batch):
            start = batch_index * self.batch_size
            end = min(self.size, start + self.batch_size)
            yield self.xs[start:end], self.ys[start:end]


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {self.std}")

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class PEMS08WindowLoader:
    """Build PEMS08 input/target windows lazily from a continuous series."""

    def __init__(
        self,
        series,
        start_indices,
        scaler,
        batch_size,
        input_len=12,
        output_len=12,
        slots_per_day=288,
        start_day_of_week=4,
    ):
        self.series = series
        self.indices = np.asarray(start_indices, dtype=np.int64)
        self.scaler = scaler
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len
        self.slots_per_day = slots_per_day
        self.start_day_of_week = start_day_of_week
        self.size = len(self.indices)
        self.num_batch = math.ceil(self.size / batch_size)

    def shuffle(self):
        np.random.shuffle(self.indices)

    def _time_features(self, time_indices):
        time_of_day = (
            time_indices % self.slots_per_day
        ).astype(np.float32) / self.slots_per_day
        day_of_week = (
            self.start_day_of_week
            + time_indices // self.slots_per_day
        ) % 7
        return time_of_day, day_of_week.astype(np.float32)

    def get_iterator(self):
        input_offsets = np.arange(self.input_len, dtype=np.int64)
        output_offsets = (
            np.arange(self.output_len, dtype=np.int64) + self.input_len
        )

        for batch_index in range(self.num_batch):
            start = batch_index * self.batch_size
            end = min(self.size, start + self.batch_size)
            window_starts = self.indices[start:end]

            input_indices = window_starts[:, None] + input_offsets[None, :]
            target_indices = window_starts[:, None] + output_offsets[None, :]

            flow = self.series[input_indices]
            time_of_day, day_of_week = self._time_features(input_indices)
            batch, history, nodes = flow.shape
            x = np.empty((batch, history, nodes, 3), dtype=np.float32)
            x[..., 0] = self.scaler.transform(flow)
            x[..., 1] = time_of_day[..., None]
            x[..., 2] = day_of_week[..., None]

            y = self.series[target_indices][..., None].astype(
                np.float32, copy=False
            )
            yield x, y


def load_pems08_dataset(
    dataset_dir,
    batch_size,
    valid_batch_size=None,
    test_batch_size=None,
    input_len=12,
    output_len=12,
    train_ratio=0.6,
    val_ratio=0.2,
    slots_per_day=288,
    start_day_of_week=4,
):
    """
    Load raw PEMS08 and split it chronologically into 60/20/20 segments.

    Windows never cross split boundaries. The scaler is fitted only on the raw
    training segment. Feature 0 is normalized flow; features 1 and 2 are
    time-of-day in [0, 1) and day-of-week in [0, 6].
    """

    dataset_dir = Path(dataset_dir)
    series_path = dataset_dir / "pems08.npz"
    adjacency_path = dataset_dir / "pems08_adj.npy"
    if not series_path.is_file() or not adjacency_path.is_file():
        raise FileNotFoundError(
            "PEMS08 requires pems08.npz and pems08_adj.npy under "
            f"{dataset_dir.resolve()}"
        )

    with np.load(series_path) as archive:
        if "data" not in archive:
            raise KeyError(f"{series_path} does not contain a 'data' array")
        raw = archive["data"]

    if raw.ndim != 3 or raw.shape[-1] < 1:
        raise ValueError(f"Unexpected PEMS08 shape: {raw.shape}")
    series = raw[..., 0].astype(np.float32)
    adjacency = np.load(adjacency_path).astype(np.float32)
    if adjacency.shape != (series.shape[1], series.shape[1]):
        raise ValueError(
            f"Series has {series.shape[1]} nodes, adjacency is {adjacency.shape}"
        )
    if not np.isfinite(series).all() or not np.isfinite(adjacency).all():
        raise ValueError("PEMS08 contains non-finite values")

    total_steps = len(series)
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))
    window_size = input_len + output_len

    def split_indices(start, end):
        count = end - start - window_size + 1
        if count <= 0:
            raise ValueError(
                f"Split [{start}, {end}) is too short for {window_size} steps"
            )
        return np.arange(start, start + count, dtype=np.int64)

    scaler = StandardScaler(
        mean=series[:train_end].mean(),
        std=series[:train_end].std(),
    )
    valid_batch_size = valid_batch_size or batch_size
    test_batch_size = test_batch_size or batch_size
    common = dict(
        series=series,
        scaler=scaler,
        input_len=input_len,
        output_len=output_len,
        slots_per_day=slots_per_day,
        start_day_of_week=start_day_of_week,
    )

    data = {
        "train_loader": PEMS08WindowLoader(
            start_indices=split_indices(0, train_end),
            batch_size=batch_size,
            **common,
        ),
        "val_loader": PEMS08WindowLoader(
            start_indices=split_indices(train_end, val_end),
            batch_size=valid_batch_size,
            **common,
        ),
        "test_loader": PEMS08WindowLoader(
            start_indices=split_indices(val_end, total_steps),
            batch_size=test_batch_size,
            **common,
        ),
        "scaler": scaler,
        "adj_mx": adjacency,
        "num_nodes": series.shape[1],
        "slots_per_day": slots_per_day,
        "split_points": (train_end, val_end, total_steps),
    }
    print(
        "PEMS08 windows - "
        f"train: {data['train_loader'].size}, "
        f"val: {data['val_loader'].size}, "
        f"test: {data['test_loader'].size}"
    )
    return data


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    """Load the original repository's pre-windowed train/val/test files."""

    data = {}
    for category in ("train", "val", "test"):
        category_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = category_data["x"].astype(np.float32)
        data["y_" + category] = category_data["y"].astype(np.float32)

    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(),
        std=data["x_train"][..., 0].std(),
    )
    for category in ("train", "val", "test"):
        data["x_" + category][..., 0] = scaler.transform(
            data["x_" + category][..., 0]
        )

    data["train_loader"] = DataLoader(
        data["x_train"], data["y_train"], batch_size
    )
    data["val_loader"] = DataLoader(
        data["x_val"], data["y_val"], valid_batch_size or batch_size
    )
    data["test_loader"] = DataLoader(
        data["x_test"], data["y_test"], test_batch_size or batch_size
    )
    data["scaler"] = scaler
    return data


def _masked_values(pred, true, mask_value):
    if mask_value is None:
        return pred, true
    mask = torch.gt(true, mask_value)
    return torch.masked_select(pred, mask), torch.masked_select(true, mask)


def MAE_torch(pred, true, mask_value=None):
    pred, true = _masked_values(pred, true, mask_value)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    pred, true = _masked_values(pred, true, mask_value)
    return torch.mean(torch.abs((true - pred) / true))


def RMSE_torch(pred, true, mask_value=None):
    pred, true = _masked_values(pred, true, mask_value)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    pred, true = _masked_values(pred, true, mask_value)
    return torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))


def metric(pred, real):
    return (
        MAE_torch(pred, real, 0).item(),
        MAPE_torch(pred, real, 0).item(),
        RMSE_torch(pred, real, 0).item(),
        WMAPE_torch(pred, real, 0).item(),
    )
