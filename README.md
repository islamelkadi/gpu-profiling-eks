# GPU Profiling with PyTorch on EKS Auto Mode

> Profile GPU training workloads on Amazon EKS to identify performance bottlenecks, optimize resource utilization, and reduce infrastructure costs.

By the end of this guide, you'll have a GPU-accelerated PyTorch training job running on Amazon EKS, producing real profiling data that tells you exactly where your model spends its time. More importantly, you'll understand *why* it spends time there — and that understanding is the foundation for every optimization technique that follows.

---

## Table of Contents

- [Why Profiling Matters](#why-profiling-matters)
- [What You Can Learn From Profiling](#what-you-can-learn-from-profiling)
- [Real-World Use Cases](#real-world-use-cases)
- [Core Concepts](#core-concepts)
  - [GPU Architecture Basics](#gpu-architecture-basics)
  - [What Is a CUDA Kernel?](#what-is-a-cuda-kernel)
  - [How PyTorch Talks to the GPU](#how-pytorch-talks-to-the-gpu)
  - [The Training Loop, Step by Step](#the-training-loop-step-by-step)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
  - [PyTorch Profiler and CUDA Instrumentation](#pytorch-profiler-and-cuda-instrumentation)
  - [Reading a Chrome Trace](#reading-a-chrome-trace)
  - [EKS Auto Mode and GPU Node Provisioning](#eks-auto-mode-and-gpu-node-provisioning)
  - [VPC Networking for Your Pods](#vpc-networking-for-your-pods)
- [Architecture](#architecture)
- [Key Decisions and Trade-offs](#key-decisions-and-trade-offs)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Interpreting Results](#interpreting-results)
- [Next Steps After Profiling](#next-steps-after-profiling)
- [Common Pitfalls](#common-pitfalls)
- [Further Reading](#further-reading)
- [Repository Structure](#repository-structure)

---

## Why Profiling Matters

Every ML optimization starts with the same question: *where is the bottleneck?*

You might assume your model is slow because the GPU isn't fast enough. But in practice, the GPU is often sitting idle — waiting for data to arrive from CPU memory, waiting for the next batch to be prepared, or waiting for a synchronization point that you didn't know existed. GPU profiling is the tool that reveals these hidden bottlenecks.

Think of it like this: if you're trying to make a car go faster, you wouldn't just bolt on a bigger engine without first checking whether the brakes are dragging, the tires are flat, or the fuel line is clogged. Profiling is the diagnostic step that tells you *what's actually slowing things down* before you start optimizing.

Without profiling, you're guessing. You might think the convolution layers are slow, but the profiler reveals that 40% of your time is spent in data loading. Or you might think the forward pass is the bottleneck, but it's actually the backward pass that dominates. Profiling replaces guessing with data.

## What You Can Learn From Profiling

| Metric | What It Tells You | Why It Matters |
|--------|-------------------|----------------|
| GPU Utilization % | How much time the GPU is actively computing vs. idle | If below 70-80%, something is starving the GPU of work |
| Memory Allocation | Peak GPU memory usage and allocation timeline | Helps you right-size instances and avoid OOM errors |
| Kernel Timing | How long each CUDA kernel takes (convolution, batch norm, etc.) | Identifies which operations are compute-bound vs. memory-bound |
| CPU-GPU Sync Points | Where the CPU waits for the GPU or vice versa | Hidden synchronization is a common source of wasted time |
| Data Loading Time | Gaps between kernel executions | Reveals if the DataLoader is the bottleneck, not the model |
| Memory Bandwidth | Rate of data transfer between CPU and GPU memory | Shows if you're PCIe-bound or memory-bound |

## Real-World Use Cases

**Right-sizing GPU instances**: Profiling shows your actual GPU memory usage and compute utilization. If you're using a `g5.2xlarge` but only using 8GB of the 24GB A10G, you can downsize to `g5.xlarge` and cut costs in half.

**Identifying data pipeline bottlenecks**: A common pattern is the GPU sitting idle between batches because the DataLoader can't prepare data fast enough. Profiling reveals this as gaps in the CUDA kernel timeline. The fix is usually increasing `num_workers` or enabling `pin_memory`.

**Choosing optimization strategies**: Profiling data tells you which optimization to apply first. If you're memory-bound, AMP (mixed precision) will help most. If you're compute-bound, `torch.compile` is the better bet. If data loading is the bottleneck, neither will help — you need to fix the pipeline.

**Capacity planning**: Before scaling to multi-GPU (DDP/FSDP), profiling a single GPU tells you the theoretical scaling efficiency. If single-GPU utilization is only 50%, adding more GPUs won't help — you'll just have more idle GPUs.

**Cost optimization**: GPU instances are expensive ($1-30+/hr). Profiling helps you understand whether you're paying for compute you're actually using, or paying for a GPU that's mostly waiting.

---

## Core Concepts

### GPU Architecture Basics

Before you can profile GPU workloads, you need a mental model of what a GPU actually is and how it differs from a CPU.

A CPU is designed for *sequential* tasks. It has a small number of powerful cores (4-64 on a typical server) that can each handle complex, branching logic quickly. A GPU is designed for *parallel* tasks. It has thousands of simpler cores that can all execute the same operation on different pieces of data simultaneously.

**CUDA Cores** are the individual processing units on an NVIDIA GPU. A single NVIDIA A10G GPU (the one inside a `g5.xlarge` instance) has 9,216 CUDA cores. Each core can perform one floating-point operation per clock cycle. When you multiply a matrix by another matrix — which is what neural networks do constantly — the GPU distributes that work across thousands of cores in parallel.

**Streaming Multiprocessors (SMs)** are groups of CUDA cores that share resources. Think of an SM as a team that works together. The A10G has 80 SMs, each containing 128 CUDA cores. An SM shares a scheduler, registers, and a small pool of fast memory.

**GPU Memory Hierarchy** is where things get interesting for performance:

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Memory Hierarchy                  │
├─────────────────────────────────────────────────────────┤
│  Registers (per thread)                                 │
│  ├── Fastest: ~1 clock cycle access                     │
│  ├── Smallest: a few KB per SM                          │
│  └── Each CUDA thread gets its own registers            │
│                                                         │
│  Shared Memory (per SM)                                 │
│  ├── Fast: ~5-10 clock cycles                           │
│  ├── Small: 48-164 KB per SM                            │
│  └── Shared across all threads in a block               │
│                                                         │
│  Global Memory (HBM — the "GPU RAM")                    │
│  ├── Slow: ~200-400 clock cycles                        │
│  ├── Large: 24 GB on A10G                               │
│  └── Accessible by all threads on the GPU               │
│      (this is where your model weights and data live)   │
└─────────────────────────────────────────────────────────┘
```

The performance gap between these levels is enormous. Accessing global memory is roughly 100x slower than accessing a register. When the profiler shows you that a kernel is "memory-bound," it usually means the GPU cores are waiting for data to arrive from global memory.

### What Is a CUDA Kernel?

A CUDA kernel is a function that runs on the GPU. When PyTorch executes an operation like `torch.matmul(A, B)`, it doesn't run that multiplication on the CPU. Instead, it launches a CUDA kernel — a GPU function that distributes the work across thousands of cores.

1. Your Python code calls `torch.matmul(A, B)`.
2. PyTorch looks up the appropriate CUDA kernel for matrix multiplication.
3. PyTorch tells the GPU: "Run this kernel on these two tensors."
4. The GPU scheduler assigns blocks of the computation to SMs.
5. Each SM distributes work across its CUDA cores.
6. The result is written back to GPU global memory.
7. Control returns to your Python code.

Steps 3-6 happen on the GPU. Step 7 is where things can get tricky — if your Python code immediately needs the result, the CPU has to *wait* for the GPU to finish. This CPU-GPU synchronization is another common source of hidden bottlenecks.

### How PyTorch Talks to the GPU

PyTorch uses the **CUDA runtime** to communicate with the GPU:

```python
# Move data to GPU
x = torch.randn(64, 3, 32, 32).cuda()

# Run a convolution (this launches a CUDA kernel)
output = model(x)
```

When you call `.cuda()`, PyTorch allocates memory on the GPU and copies the tensor data from CPU RAM to GPU global memory. This copy goes over the PCIe bus and takes real time — the profiler will show you exactly how much.

An important detail: **PyTorch launches CUDA kernels asynchronously**. When you call `model(x)`, PyTorch doesn't wait for the GPU to finish before moving to the next Python line. It queues the kernel launch and moves on. This is great for performance but it means that naive timing with `time.time()` will give you wrong numbers. The profiler handles this correctly by synchronizing with the GPU before measuring.

### The Training Loop, Step by Step

Here's what happens at the hardware level during each training iteration — the profiler will show you each of these phases:

```python
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()    # Phase 1: Data Transfer
    output = model(data)                          # Phase 2: Forward Pass
    loss = criterion(output, target)              # Phase 3: Loss Computation
    optimizer.zero_grad()
    loss.backward()                               # Phase 4: Backward Pass
    optimizer.step()                              # Phase 5: Optimizer Step
```

**Phase 1 — Data Transfer**: The CPU loads a batch from disk/RAM, applies transforms, and copies to GPU memory. If the DataLoader is slow, the GPU sits idle here.

**Phase 2 — Forward Pass**: Input flows through the model layer by layer. Each layer launches one or more CUDA kernels.

**Phase 3 — Loss Computation**: Model output is compared to ground truth. Usually fast — a single kernel on a small tensor.

**Phase 4 — Backward Pass**: Gradients are computed by walking backward through the computation graph. Typically takes 2-3x longer than the forward pass.

**Phase 5 — Optimizer Step**: Parameters are updated using computed gradients. Adam is more expensive than SGD because it maintains running averages.

The profiler will show you exactly how much time is spent in each phase. A common surprise: the DataLoader (Phase 1) is often the bottleneck, not the GPU computation.

---

## How It Works Under the Hood

### PyTorch Profiler and CUDA Instrumentation

PyTorch Profiler (`torch.profiler`) instruments your training loop to capture detailed performance data:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (data, target) in enumerate(train_loader):
        # ... training step ...
        prof.step()
```

- **`activities=[CPU, CUDA]`**: Records both CPU and CUDA operations for the full picture.
- **`schedule(wait=1, warmup=1, active=3)`**: Skips the first iteration, warms up for one, then records 3 iterations. Avoids capturing one-time initialization costs.
- **`record_shapes=True`**: Records tensor shapes to understand why a kernel is slow.
- **`profile_memory=True`**: Tracks GPU memory allocations over time.
- **`with_stack=True`**: Records Python call stacks so you can trace a slow kernel back to the exact line of code.

Under the hood, the profiler uses NVIDIA's CUPTI (CUDA Profiling Tools Interface) to intercept CUDA API calls with nanosecond precision.

### Reading a Chrome Trace

The profiler outputs a trace file viewable in Chrome's trace viewer (`chrome://tracing`) or TensorBoard:

```
Time →
┌──────────────────────────────────────────────────────────────┐
│ CPU Thread                                                    │
│ ┌──────┐ ┌──────┐ ┌──────────────┐ ┌──────┐ ┌────────────┐ │
│ │DataLd│ │ .cuda│ │  model(x)    │ │ loss │ │ backward() │ │
│ └──────┘ └──────┘ └──────────────┘ └──────┘ └────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ CUDA Stream                                                   │
│          ┌────┐ ┌───┐┌────┐┌───┐┌──┐ ┌────────────────────┐ │
│          │HtoD│ │cv1││bn1 ││rl1││ce│ │   backward kernels │ │
│          └────┘ └───┘└────┘└───┘└──┘ └────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ GPU Memory                                                    │
│          ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲          │
│    ─────╱  activations accumulate during forward   ╲─────    │
│                                                      ↓       │
│                                    freed during backward     │
└──────────────────────────────────────────────────────────────┘
```

The most valuable thing to look for: **gaps**. A gap in the CUDA stream means the GPU is idle. A gap in the CPU thread means the CPU is blocked. These gaps are your optimization targets.

### EKS Auto Mode and GPU Node Provisioning

Amazon EKS Auto Mode handles node management entirely. Here's what happens when you submit a GPU training Job:

1. You submit a Job requesting `nvidia.com/gpu: 1`.
2. The Kubernetes scheduler sees no existing node can satisfy this.
3. Karpenter (running under Auto Mode) detects the unschedulable pod.
4. Karpenter selects the cheapest EC2 instance type that fits — `g5.xlarge` (1 NVIDIA A10G, 4 vCPUs, 16 GB RAM).
5. Karpenter launches the instance with a GPU-compatible AMI (NVIDIA drivers pre-installed).
6. The instance joins the cluster, the scheduler places your pod.
7. Your training Job runs on the GPU.
8. When the Job completes and no other pods need the node, Karpenter terminates the instance.

This takes 2-5 minutes from Job submission to pod running.

### VPC Networking for Your Pods

Your GPU pod needs network connectivity to pull container images from ECR and download datasets. The networking works through:

- **Private subnets**: Where EKS worker nodes and pods live. No direct internet access — traffic goes through a NAT Gateway.
- **Public subnets**: Where the NAT Gateway sits. Provides outbound internet access for pulling external images (NVIDIA NGC) and datasets.
- **VPC Endpoints**: Private connectivity to AWS services (ECR, S3, EKS, STS, SSM, CloudWatch, EFS) so that traffic stays within AWS and doesn't traverse the NAT Gateway.
- **NAT Gateway**: Only needed for external traffic (NVIDIA NGC images, CIFAR-10 download from `cs.toronto.edu`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Account                          │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  VPC (10.0.0.0/16)                    │  │
│  │                                                       │  │
│  │  Private Subnets                Public Subnets        │  │
│  │  ┌─────────────────┐           ┌──────────────┐      │  │
│  │  │ EKS Auto Mode   │           │ NAT Gateway  │      │  │
│  │  │ ┌─────────────┐ │           └──────────────┘      │  │
│  │  │ │ GPU Node    │ │                                  │  │
│  │  │ │ (g5.xlarge) │ │  VPC Endpoints:                  │  │
│  │  │ │  ┌────────┐ │ │  EKS, ECR, S3, STS,             │  │
│  │  │ │  │  Pod   │ │ │  SSM, CloudWatch, EFS            │  │
│  │  │ │  │ train  │ │ │                                  │  │
│  │  │ │  │  .py   │ │ │  ┌──────────────┐               │  │
│  │  │ │  └───┬────┘ │ │  │ SSM Bastion  │               │  │
│  │  │ │      │      │ │  │ (t3.micro)   │               │  │
│  │  │ │      ▼      │ │  └──────────────┘               │  │
│  │  │ │  EFS Mount  │ │                                  │  │
│  │  │ │  /data      │ │                                  │  │
│  │  │ └─────────────┘ │                                  │  │
│  │  └─────────────────┘                                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────┐  ┌──────┐  ┌──────┐                              │
│  │ ECR  │  │ EFS  │  │ KMS  │                              │
│  └──────┘  └──────┘  └──────┘                              │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:**

1. `make infra` → Terraform provisions VPC, EKS (Auto Mode), EFS, ECR, bastion, VPC endpoints.
2. `make build` → Container image built locally with finch/docker.
3. `make push` → Image pushed to ECR.
4. `make run` → K8s Job submitted. Karpenter provisions a `g5.xlarge` GPU node.
5. `train.py` runs: downloads CIFAR-10 to EFS, trains ResNet-18, profiles epoch 1.
6. `make logs` → Profiling output (structured JSON) retrieved from the pod.

## Key Decisions and Trade-offs

**EKS Auto Mode vs. Self-Managed Node Groups vs. Karpenter**

| Approach | You Manage | Pros | Cons |
|----------|-----------|------|------|
| Self-Managed Node Groups | AMI, launch template, ASG, NVIDIA drivers, device plugin | Full control | Lots of undifferentiated work |
| Karpenter (self-installed) | Karpenter Helm chart, NodePool CRDs, EC2NodeClass | Flexible, fast scaling | You own the Karpenter lifecycle |
| EKS Auto Mode | Nothing — AWS manages it all | Zero node management, GPU-ready out of the box | Less customization |

**terraform-aws-modules vs. Raw Resources**: The VPC module handles ~20 resources in a single module call. Writing these as raw resources would be 200+ lines. The trade-off is abstraction — when something goes wrong, you need to understand what the module does under the hood.

**ResNet-18 on CIFAR-10**: Small enough to iterate fast (11.7M parameters, trains in minutes), complex enough to show real GPU behavior (convolutions, batch norm, skip connections, ReLU). This is a profiling exercise, not a model accuracy exercise.

---

## Getting Started

### Prerequisites

- AWS CLI configured with credentials
- Terraform >= 1.5
- finch (or docker)
- kubectl
- make
- `envsubst` (ships with most Linux/macOS)

### End-to-End

```bash
# Full pipeline: provision infra → build image → push to ECR → deploy job
make all

# Or step by step:
make setup      # ECR login, kubectl context, terraform init
make infra      # Provision VPC, EKS, EFS, ECR, bastion
make storage    # Deploy EFS StorageClass and PVC
make build      # Build the training container image
make push       # Push to ECR
make run        # Submit the profiling Job
make logs       # Tail the output
make teardown   # Destroy everything
```

### SSM into the Bastion (for kubectl)

The EKS API is private-only. Use the SSM bastion to run kubectl:

```bash
INSTANCE_ID=$(cd terraform && terraform output -raw bastion_instance_id)
aws ssm start-session --target $INSTANCE_ID

# Inside the session:
aws eks update-kubeconfig --name ai-infra-dev --region us-west-2
kubectl get pods -l app=training
```

## Configuration

Training parameters are defined in `config.yaml`:

```yaml
model: resnet18
dataset: cifar10
batch_size: 128
epochs: 5
learning_rate: 0.1
momentum: 0.9
weight_decay: 0.0005
num_workers: 2
data_dir: /data
scheduler: cosine
```

Override via CLI: `make run EPOCHS=10 BATCH_SIZE=256`

Or pass a custom config: `python -m src.training.train --config config.yaml --epochs 10`

CLI flags take precedence over the config file, which takes precedence over built-in defaults.

## Interpreting Results

The profiling Job outputs structured JSON to stdout (captured via `make logs`):

```json
{
  "stage": "gpu-profiling",
  "phase": "profiling",
  "metrics": {
    "total_cuda_time_ms": 1250.5,
    "total_cpu_time_ms": 1800.3,
    "gpu_utilization_pct": 69.5,
    "kernel_count": 4200,
    "top_kernels": [
      {"name": "volta_scudnn_128x64_relu_small_nn_v1", "cuda_time_ms": 320.1, "calls": 390},
      {"name": "void at::native::vectorized_elementwise_kernel<...>", "cuda_time_ms": 180.5, "calls": 780}
    ]
  }
}
```

Key things to look for:

- `gpu_utilization_pct` below 70% → GPU is underutilized, likely a data loading or CPU bottleneck
- Large gap between `total_cpu_time_ms` and `total_cuda_time_ms` → CPU-GPU synchronization overhead
- Top kernels dominated by memory operations → model is memory-bound, AMP would help
- Top kernels dominated by compute operations → model is compute-bound, `torch.compile` would help
- `throughput_samples_per_sec` → your baseline number, every optimization is measured against this
- `gpu_memory_peak_mb` → if close to the 24 GB limit (A10G), reduce batch size or use memory-saving techniques

## Next Steps After Profiling

Once you have profiling data, the natural optimization path is:

1. **Fix data loading** if the GPU is idle between batches (increase `num_workers`, enable `pin_memory`)
2. **Apply AMP (mixed precision)** if you're memory-bound — cuts memory usage ~50%, often doubles throughput
3. **Apply `torch.compile`** if you're compute-bound — fuses operations and generates optimized CUDA kernels
4. **Scale to multi-GPU** (DDP/FSDP) once single-GPU utilization is above 80%
5. **Right-size your instance** based on actual memory and compute usage from the profiling data

---

## Common Pitfalls

### GPU Node Not Provisioning

**Symptom**: Pod stays in `Pending` for more than 10 minutes.

**Causes**: Service quota limit on GPU instances, AZ capacity, missing subnet tags.

**Fix**: Check pod events and Karpenter logs:
```bash
kubectl describe pod <pod-name>
kubectl logs -n kube-system -l app.kubernetes.io/name=karpenter
```

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`.

**Cause**: Model + batch + optimizer states + activations don't fit in GPU memory.

**Fix**: Reduce batch size. The profiler's memory tracking shows exactly where the peak occurs.

### Profiler Output Too Large

**Symptom**: Chrome trace file is hundreds of MB and Chrome crashes loading it.

**Fix**: The profiler schedule limits recording to 3 iterations after warmup. If still too large, reduce `active` in the schedule.

### `kubectl` Not Configured

**Symptom**: `kubectl get nodes` returns an error.

**Fix**: The EKS API is private-only. SSM into the bastion first:
```bash
aws ssm start-session --target <instance-id>
aws eks update-kubeconfig --name ai-infra-dev --region us-west-2
```

### ECR Authentication Expired

**Symptom**: `docker push` fails with "no basic auth credentials" or pod shows `ImagePullBackOff`.

**Fix**: ECR tokens expire after 12 hours. Re-authenticate:
```bash
aws ecr get-login-password --region us-west-2 | finch login --username AWS --password-stdin <ecr-registry>
```

---

## Further Reading

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html) — Official reference for `torch.profiler`. Focus on the "Getting Started" tutorial and the `schedule` API.
- [PyTorch Profiler TensorBoard Plugin](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) — Visualizing profiler output in TensorBoard. More user-friendly than raw Chrome traces.
- [NVIDIA GPU Architecture Whitepaper (Ampere)](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf) — Deep dive into the A10G's architecture. Focus on Streaming Multiprocessors and memory hierarchy.
- [Amazon EKS Auto Mode Documentation](https://docs.aws.amazon.com/eks/latest/userguide/automode.html) — How Auto Mode selects instance types and handles GPU workloads.
- [Szymon Migacz — "PyTorch Performance Tuning Guide" (GTC 2021)](https://www.youtube.com/watch?v=9mS1fIYj1So) — 30-minute talk covering profiling, AMP, DataLoader tuning, and common pitfalls. A great preview of optimization techniques you can apply to this baseline.
- [terraform-aws-modules/eks/aws](https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest) — The Terraform module used for EKS. Read the "Auto Mode" section.
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Focus on Chapter 2 (Programming Model) and Chapter 5 (Memory Hierarchy).

---

## Repository Structure

```
├── terraform/
│   ├── provider.tf         # AWS provider + backend
│   ├── main.tf             # VPC, VPC endpoints, EKS Auto Mode
│   ├── efs.tf              # EFS for training data persistence
│   ├── ecr.tf              # ECR repository for container images
│   ├── bastion.tf          # SSM bastion for private kubectl access
│   ├── variables.tf        # Input variables
│   ├── outputs.tf          # Terraform outputs
│   ├── locals.tf           # Local values
│   └── params/input.tfvars # Environment-specific values
├── src/
│   ├── __init__.py
│   └── training/
│       ├── __init__.py
│       ├── train.py        # Entrypoint — training loop with profiling
│       ├── data.py         # CIFAR-10 data loaders and transforms
│       ├── model.py        # ResNet-18 adapted for CIFAR-10
│       ├── engine.py       # Train/eval/profiled epoch loops
│       └── profiler.py     # Profiler metric extraction
├── k8s/
│   ├── training/
│   │   └── profiling-job.yaml
│   └── storage/
│       ├── efs-storageclass.yaml
│       └── efs-pvc.yaml
├── scripts/
│   ├── setup.sh
│   └── teardown.sh
├── config.yaml             # Training configuration (YAML)
├── Dockerfile
├── Makefile
├── requirements.txt
├── REPORT.md               # Profiling results template
└── README.md
```
