"""Training and evaluation loops with optional profiling."""

import time

import torch
from torch.profiler import ProfilerActivity, profile, schedule


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Run one training epoch. Returns (avg_loss, throughput, elapsed)."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    epoch_start = time.perf_counter()

    for _, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    elapsed = time.perf_counter() - epoch_start
    avg_loss = running_loss / total_samples
    throughput = total_samples / elapsed if elapsed > 0 else 0.0
    return avg_loss, throughput, elapsed


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def run_profiled_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Run a single profiled training epoch. Returns (prof, avg_loss, throughput, elapsed)."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    epoch_start = time.perf_counter()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            prof.step()

    elapsed = time.perf_counter() - epoch_start
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    throughput = total_samples / elapsed if elapsed > 0 else 0.0
    return prof, avg_loss, throughput, elapsed
