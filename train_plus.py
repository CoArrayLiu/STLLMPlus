import argparse
import csv
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import numpy as np
import torch

import util
from model_ST_LLM_plus import ST_LLM
from ranger21 import Ranger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Llama 3.1 ST-LLM+ on PEMS08"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--data_dir", default="./data/st_data/pems08"
    )
    parser.add_argument(
        "--model_path", default="./Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--save_dir", default="./logs/llama31_8b_pems08"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument(
        "--llm_layers",
        type=int,
        default=32,
        help="Number of Llama layers to retain; 0 uses all 32",
    )
    parser.add_argument(
        "--graph_layers",
        type=int,
        default=2,
        help=(
            "Final U Llama layers using graph-masked, unfrozen attention; "
            "their FFNs stay frozen"
        ),
    )
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument(
        "--amp_dtype",
        choices=("bf16", "fp16", "none"),
        default="bf16",
    )
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument(
        "--disable_gradient_checkpointing", action="store_true"
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Limit batches per epoch for smoke tests; 0 means all",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="Limit validation/test batches for smoke tests; 0 means all",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scaler,
        device,
        amp_dtype,
        grad_accum_steps,
        clip_grad,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad = clip_grad

        if amp_dtype == "bf16":
            if device.type == "cuda" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("This CUDA device does not support BF16")
            self.autocast_dtype = torch.bfloat16
        elif amp_dtype == "fp16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = None

        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda" and amp_dtype == "fp16",
        )
        self.trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

    def autocast(self):
        if self.device.type != "cuda" or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type="cuda", dtype=self.autocast_dtype
        )

    def _batch_limit(self, loader, requested_limit):
        if requested_limit and requested_limit > 0:
            return min(loader.num_batch, requested_limit)
        return loader.num_batch

    def train_epoch(self, loader, max_batches=0):
        self.model.train()
        loader.shuffle()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_samples = 0
        batch_limit = self._batch_limit(loader, max_batches)

        for batch_index, (x, y) in enumerate(loader.get_iterator()):
            if batch_index >= batch_limit:
                break
            inputs = torch.from_numpy(x).to(
                self.device, non_blocking=True
            )
            target = torch.from_numpy(y[..., 0]).to(
                self.device, non_blocking=True
            )

            with self.autocast():
                prediction_norm = self.model(inputs).squeeze(-1)
                prediction = self.scaler.inverse_transform(prediction_norm)
                loss = util.MAE_torch(prediction, target, 0.0)
                scaled_loss = loss / self.grad_accum_steps

            self.grad_scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % self.grad_accum_steps == 0
                or batch_index + 1 == batch_limit
            )
            if should_step:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters, self.clip_grad
                )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            batch_samples = len(x)
            total_loss += loss.detach().float().item() * batch_samples
            total_samples += batch_samples

        return total_loss / max(total_samples, 1)

    @torch.inference_mode()
    def evaluate(self, loader, max_batches=0):
        self.model.eval()
        batch_limit = self._batch_limit(loader, max_batches)
        horizon_count = None
        horizon_abs = None
        horizon_squared = None
        horizon_ape = None
        horizon_target_abs = None

        for batch_index, (x, y) in enumerate(loader.get_iterator()):
            if batch_index >= batch_limit:
                break
            inputs = torch.from_numpy(x).to(
                self.device, non_blocking=True
            )
            target = torch.from_numpy(y[..., 0]).to(
                self.device, non_blocking=True
            )
            with self.autocast():
                prediction_norm = self.model(inputs).squeeze(-1)
            prediction = self.scaler.inverse_transform(
                prediction_norm.float()
            )

            mask = target > 0
            absolute = torch.abs(prediction - target)
            squared = (prediction - target) ** 2
            ape = torch.where(
                mask,
                absolute / target.clamp_min(1e-6),
                torch.zeros_like(absolute),
            )
            reduce_dims = (0, 2)
            values = (
                mask.sum(dim=reduce_dims).double(),
                (absolute * mask).sum(dim=reduce_dims).double(),
                (squared * mask).sum(dim=reduce_dims).double(),
                ape.sum(dim=reduce_dims).double(),
                (target.abs() * mask).sum(dim=reduce_dims).double(),
            )
            if horizon_count is None:
                (
                    horizon_count,
                    horizon_abs,
                    horizon_squared,
                    horizon_ape,
                    horizon_target_abs,
                ) = values
            else:
                horizon_count += values[0]
                horizon_abs += values[1]
                horizon_squared += values[2]
                horizon_ape += values[3]
                horizon_target_abs += values[4]

        if horizon_count is None:
            raise RuntimeError("Evaluation loader produced no batches")

        horizon_count = horizon_count.clamp_min(1)
        per_horizon = {
            "mae": (horizon_abs / horizon_count).cpu().numpy(),
            "rmse": torch.sqrt(
                horizon_squared / horizon_count
            ).cpu().numpy(),
            "mape": (horizon_ape / horizon_count).cpu().numpy(),
            "wmape": (
                horizon_abs / horizon_target_abs.clamp_min(1e-12)
            ).cpu().numpy(),
        }
        total_count = horizon_count.sum()
        overall = {
            "mae": (horizon_abs.sum() / total_count).item(),
            "rmse": torch.sqrt(
                horizon_squared.sum() / total_count
            ).item(),
            "mape": (horizon_ape.sum() / total_count).item(),
            "wmape": (
                horizon_abs.sum()
                / horizon_target_abs.sum().clamp_min(1e-12)
            ).item(),
        }
        return overall, per_horizon


def save_checkpoint(path, model, args, epoch, val_metrics):
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "trainable_state_dict": trainable_state,
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def load_trainable_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["trainable_state_dict"]
    model_keys = dict(model.named_parameters())
    unexpected = sorted(set(state) - set(model_keys))
    if unexpected:
        raise RuntimeError(
            f"Checkpoint contains unknown trainable keys: {unexpected[:5]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            model_keys[name].copy_(
                value.to(
                    device=model_keys[name].device,
                    dtype=model_keys[name].dtype,
                )
            )
    return checkpoint


def append_history(path, row):
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset = util.load_pems08_dataset(
        dataset_dir=args.data_dir,
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_trainable.pth"
    history_path = save_dir / "train.csv"

    print(args)
    print(
        f"Loading Llama 3.1 from {Path(args.model_path).resolve()}",
        flush=True,
    )
    model = ST_LLM(
        adj_mx=dataset["adj_mx"],
        model_path=args.model_path,
        input_dim=3,
        num_nodes=dataset["num_nodes"],
        input_len=args.input_len,
        output_len=args.output_len,
        llm_layers=args.llm_layers,
        graph_layers=args.graph_layers,
        embedding_dim=args.embedding_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dropout=args.dropout,
        slots_per_day=dataset["slots_per_day"],
        gradient_checkpointing=not args.disable_gradient_checkpointing,
    )
    model.to(device)

    total_parameters = model.param_num()
    trainable_parameters = model.count_trainable_params()
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(
        f"Trainable ratio: {100 * trainable_parameters / total_parameters:.4f}%"
    )

    optimizer = Ranger(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=dataset["scaler"],
        device=device,
        amp_dtype=args.amp_dtype,
        grad_accum_steps=args.grad_accum_steps,
        clip_grad=args.clip_grad,
    )

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_started = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_mae = trainer.train_epoch(
            dataset["train_loader"], args.max_train_batches
        )
        val_metrics, _ = trainer.evaluate(
            dataset["val_loader"], args.max_eval_batches
        )
        elapsed = time.time() - epoch_started

        row = {
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "val_wmape": val_metrics["wmape"],
            "seconds": elapsed,
        }
        append_history(history_path, row)
        print(
            f"Epoch {epoch:03d} | train MAE {train_mae:.4f} | "
            f"val MAE {val_metrics['mae']:.4f} | "
            f"RMSE {val_metrics['rmse']:.4f} | "
            f"MAPE {val_metrics['mape']:.4f} | "
            f"WMAPE {val_metrics['wmape']:.4f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path, model, args, epoch, val_metrics
            )
            print(
                f"Saved validation-best trainable weights: {checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {args.patience} "
                    "epochs without validation improvement"
                )
                break

    checkpoint = load_trainable_checkpoint(checkpoint_path, model)
    test_metrics, horizon_metrics = trainer.evaluate(
        dataset["test_loader"], args.max_eval_batches
    )

    print(
        f"Best validation epoch: {best_epoch} "
        f"(MAE {best_val_mae:.4f})"
    )
    for horizon in range(args.output_len):
        print(
            f"Horizon {horizon + 1:02d} | "
            f"MAE {horizon_metrics['mae'][horizon]:.4f} | "
            f"RMSE {horizon_metrics['rmse'][horizon]:.4f} | "
            f"MAPE {horizon_metrics['mape'][horizon]:.4f} | "
            f"WMAPE {horizon_metrics['wmape'][horizon]:.4f}"
        )
    print(
        "PEMS08 test average | "
        f"MAE {test_metrics['mae']:.4f} | "
        f"RMSE {test_metrics['rmse']:.4f} | "
        f"MAPE {test_metrics['mape']:.4f} | "
        f"WMAPE {test_metrics['wmape']:.4f}"
    )
    print(
        f"Total training time: {time.time() - training_started:.1f}s; "
        f"loaded checkpoint epoch {checkpoint['epoch']}"
    )


if __name__ == "__main__":
    main()
