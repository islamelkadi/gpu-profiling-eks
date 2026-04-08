# Baseline + Optimized PyTorch training
# Base image: AWS Deep Learning Container (PyTorch 2.5.1, CUDA 12.4, Python 3.11)
# Pre-installed: torch, torchvision, numpy, CUDA toolkit, cudNN, NCCL
From 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir \
    "ai-infra-common @ git+https://github.com/islamelkadi/ai-common-infra-utils.git@main"

COPY src/ src/

ENTRYPOINT ["python", "-m", "src.training.train"]
CMD ["--epochs", "5", "--batch-size", "128"]
