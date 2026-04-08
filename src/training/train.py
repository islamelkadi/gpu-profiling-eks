#!/usr/bin/env python3
"""Baseline ResNet-18 training on CIFAR-10 with PyTorch Profiler.

Usage:
    python -m src.training.train --batch-size 128 --epochs 5
    python -m src.training.train --config config.yaml
"""

import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ai_infra_common.cli import add_training_args
from ai_infra_common.logging import log_metric
from ai_infra_common.metrics import GPUMetricsCollector

from src.training.data import get_data_loaders
from src.training.model import build_model
from src.training.engine import train_one_epoch, evaluate, run_profiled_epoch
from src.training.profiler import extract_profiler_metrics

logger = logging.getLogger(__name__)

STAGE = "gpu-profiling"
SCRIPT = "train.py"

# Default training configuration
DEFAULTS = {
    "model": "resnet18",
    "dataset": "cifar10",
    "batch_size": 128,
    "epochs": 5,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "num_workers": 2,
    "data_dir": "./data",
    "scheduler": "cosine",
}


def load_config(config_path=None):
    """Load config from YAML file, falling back to defaults."""
    config = dict(DEFAULTS)
    if config_path:
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        config.update(overrides)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResNet-18 training on CIFAR-10 with PyTorch Profiler"
    )
    add_training_args(parser)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides CLI defaults)")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected, falling back to CPU")
    return device


def build_run_config(cfg, device):
    run_config = {
        "model": cfg["model"], "dataset": cfg["dataset"],
        "batch_size": cfg["batch_size"], "epochs": cfg["epochs"],
        "optimizer": "sgd", "learning_rate": cfg["learning_rate"],
        "momentum": cfg["momentum"], "weight_decay": cfg["weight_decay"],
        "num_workers": cfg["num_workers"], "scheduler": cfg["scheduler"],
        "accelerator": "nvidia-gpu" if device.type == "cuda" else "cpu",
    }
    if device.type == "cuda":
        run_config["gpu_name"] = torch.cuda.get_device_name(0)
    return run_config


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    args = parse_args()
    cfg = load_config(args.config)

    # CLI args override config file
    for key in ["batch_size", "epochs", "learning_rate", "num_workers", "data_dir"]:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val

    device = get_device()
    run_config = build_run_config(cfg, device)
    gpu = GPUMetricsCollector()

    log_metric(stage=STAGE, script=SCRIPT, phase="startup", metrics={"device": device.type}, config=run_config)

    logger.info("Loading %s dataset from %s...", cfg["dataset"], cfg["data_dir"])
    train_loader, test_loader = get_data_loaders(args)
    logger.info("Loaded %d train, %d test samples", len(train_loader.dataset), len(test_loader.dataset))

    logger.info("Building %s model...", cfg["model"])
    model = build_model(args, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg["learning_rate"],
                          momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    gpu.reset_peak_stats()
    t0 = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        logger.info("Epoch %d/%d", epoch, cfg["epochs"])

        if epoch == 1:
            prof, loss, tput, elapsed = run_profiled_epoch(model, train_loader, criterion, optimizer, device)
            prof_metrics = extract_profiler_metrics(prof, device)
            log_metric(stage=STAGE, script=SCRIPT, phase="profiling",
                       metrics={"epoch": epoch, **prof_metrics, **gpu.to_dict()}, config=run_config)
            sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
            logger.info("Profiler Summary:\n%s", prof.key_averages().table(sort_by=sort_key, row_limit=15))
        else:
            loss, tput, elapsed = train_one_epoch(model, train_loader, criterion, optimizer, device)

        test_loss, acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        log_metric(stage=STAGE, script=SCRIPT, phase="training", metrics={
            "epoch": epoch, "training_loss": round(loss, 4), "test_loss": round(test_loss, 4),
            "test_accuracy_pct": round(acc, 2), "throughput_samples_per_sec": round(tput, 1),
            "elapsed_seconds": round(elapsed, 2), **gpu.to_dict(),
        }, config=run_config)

        logger.info("loss=%.4f acc=%.2f%% tput=%.1f samples/sec time=%.2fs", loss, acc, tput, elapsed)

    total = time.perf_counter() - t0
    log_metric(stage=STAGE, script=SCRIPT, phase="complete",
               metrics={"total_epochs": cfg["epochs"], "total_elapsed_seconds": round(total, 2), **gpu.to_dict()},
               config=run_config)
    logger.info("Done: %d epochs in %.2fs", cfg["epochs"], total)


if __name__ == "__main__":
    main()
