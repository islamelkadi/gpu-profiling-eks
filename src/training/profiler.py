"""PyTorch Profiler metric extraction utilities."""

def _cuda_time(event):
    """Get CUDA/device time, compatible with PyTorch 2.5.x and 2.6+."""
    return getattr(event, "cuda_total_time", None) or getattr(event, "device_total_time", 0)

def extract_profiler_metrics(prof, device):
    """Extract key metrics from a PyTorch Profiler instance."""
    metrics = {}
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    key_averages = prof.key_averages()
    if not key_averages:
        return metrics

    total_cuda_time_ms = 0.0
    total_cpu_time_ms = 0.0
    kernel_count = 0
    for event in key_averages:
        total_cpu_time_ms += event.cpu_time_total / 1000.0
        if device.type == "cuda":
            total_cuda_time_ms += _cuda_time(event) / 1000.0
        kernel_count += event.count

    metrics["total_cpu_time_ms"] = round(total_cpu_time_ms, 2)
    metrics["kernel_count"] = kernel_count

    if device.type == "cuda":
        metrics["total_cuda_time_ms"] = round(total_cuda_time_ms, 2)
        if total_cpu_time_ms > 0:
            metrics["gpu_utilization_pct"] = round(
                min(100.0, (total_cuda_time_ms / total_cpu_time_ms) * 100), 1
            )

    for event in key_averages:
        if hasattr(event, "cpu_memory_usage") and event.cpu_memory_usage > 0:
            metrics.setdefault("cpu_memory_usage_mb", 0)
            metrics["cpu_memory_usage_mb"] += event.cpu_memory_usage / (1024 * 1024)
        if device.type == "cuda" and hasattr(event, "cuda_memory_usage") and event.cuda_memory_usage > 0:
            metrics.setdefault("cuda_memory_usage_mb", 0)
            metrics["cuda_memory_usage_mb"] += event.cuda_memory_usage / (1024 * 1024)

    for key in ("cpu_memory_usage_mb", "cuda_memory_usage_mb"):
        if key in metrics:
            metrics[key] = round(metrics[key], 2)

    sorted_events = sorted(key_averages, key=lambda e: getattr(e, sort_key, 0), reverse=True)
    top_kernels = []
    for event in sorted_events[:5]:
        kernel_info = {"name": event.key, "cpu_time_ms": round(event.cpu_time_total / 1000.0, 3), "calls": event.count}
        if device.type == "cuda":
            kernel_info["cuda_time_ms"] = round(_cuda_time(event) / 1000.0, 3)
        top_kernels.append(kernel_info)
    metrics["top_kernels"] = top_kernels

    return metrics
