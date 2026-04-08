# =============================================================================
# GPU Profiling on EKS
# =============================================================================
# Usage:
#   make help
#   make bootstrap              # one-time: install tools + plugins + venv
#   make all                    # setup → infra → storage → build → push → run
#   make run EPOCHS=10          # override training params
#   make ssm-tunnel-start       # tunnel to private EKS API via bastion
#   make ssm-tunnel-stop        # stop the tunnel
#   make teardown               # destroy everything
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# -----------------------------------------------------------------------------
# Colors (used via printf for cross-platform compatibility)
# -----------------------------------------------------------------------------
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
CYAN   := \033[0;36m
NC     := \033[0m

# -----------------------------------------------------------------------------
# Path Variables
# -----------------------------------------------------------------------------
TF_DIR       := terraform
K8S_DIR      := k8s
SRC_DIR      := src
SCRIPTS_DIR  := scripts

# -----------------------------------------------------------------------------
# AWS Configuration
# -----------------------------------------------------------------------------
AWS_REGION   ?= us-west-2
AWS_ACCOUNT  := $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
CLUSTER_NAME ?= ai-infra-dev
ENV          ?= dev

# -----------------------------------------------------------------------------
# Python / Virtual Environment
# -----------------------------------------------------------------------------
VENV_DIR     := .venv
PYTHON3      := $(shell command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || echo python3)
PYTHON       := $(VENV_DIR)/bin/python
PIP          := $(VENV_DIR)/bin/pip

# -----------------------------------------------------------------------------
# Container Runtime
# -----------------------------------------------------------------------------
RUNTIME      ?= finch

# -----------------------------------------------------------------------------
# Container Image Configuration
# -----------------------------------------------------------------------------
DLC_REGISTRY := 763104351884.dkr.ecr.$(AWS_REGION).amazonaws.com
export ECR_REGISTRY  := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com
export IMAGE_NAME    := ai-infra-training
export IMAGE_TAG     := latest

# -----------------------------------------------------------------------------
# Training Parameters (override via CLI: make run EPOCHS=10 BATCH_SIZE=256)
# -----------------------------------------------------------------------------
export EPOCHS        ?= 5
export BATCH_SIZE    ?= 128
export MEMORY        ?= 8Gi
export JOB_NAME		 ?= profiling-baseline
export EFS_ID        ?= $(shell cd $(TF_DIR) && terraform output -raw efs_id 2>/dev/null)

# -----------------------------------------------------------------------------
# GPU NodePool Limits (override via CLI: make gpu-nodepool GPU_NODEPOOL_GPUS=2)
# -----------------------------------------------------------------------------
export GPU_NODEPOOL_CPU	   ?= 8
export GPU_NODEPOOL_MEMORY ?= 32Gi
export GPU_NODEPOOL_GPUS   ?= 1

# -----------------------------------------------------------------------------
# Manifest Paths
# -----------------------------------------------------------------------------
JOB_MANIFEST      := $(K8S_DIR)/training/profiling-job.yaml
STORAGE_MANIFESTS := $(K8S_DIR)/storage

# -----------------------------------------------------------------------------
# Phony Targets
# -----------------------------------------------------------------------------
.PHONY: help bootstrap check-prereqs venv setup \
        infra storage \
        build push \
        run logs status \
        ssm-tunnel-start ssm-tunnel-stop ssm-tunnel-status \
        clean clean-venv teardown all

# =============================================================================
# Help
# =============================================================================

## help: Print available targets with descriptions
help:
	@echo ""
	@echo "GPU Profiling on EKS"
	@echo ""
	@echo "Usage: make <target> [EPOCHS=5] [BATCH_SIZE=128]"
	@echo ""
	@awk '/^## / { sub(/^## /, ""); split($$0, a, ": "); printf "  $(CYAN)%-15s$(NC) %s\n", a[1], a[2] }' $(MAKEFILE_LIST)
	@echo ""

# =============================================================================
# Bootstrap & Prerequisites (one-time setup)
# =============================================================================

## bootstrap: Install all CLI tools, plugins, and Python deps
bootstrap:
	@printf "$(GREEN)[1/5] Installing CLI tools via Homebrew...$(NC)\n"
	brew install terraform awscli kubectl helm
	@echo ""
	@printf "$(GREEN)[2/5] Installing AWS Session Manager plugin...$(NC)\n"
	brew install --cask session-manager-plugin
	@echo ""
	@printf "$(GREEN)[3/5] Creating Python virtual environment...$(NC)\n"
	$(PYTHON3) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@printf "$(GREEN)[4/5] Setting default AWS region...$(NC)\n"
	aws configure set region $(AWS_REGION)
	@echo ""
	@printf "$(GREEN)[5/5] Verifying installations...$(NC)\n"
	@terraform --version | head -1
	@aws --version
	@kubectl version --client 2>/dev/null | head -1
	@helm version --short
	@session-manager-plugin --version 2>/dev/null || printf "$(YELLOW)Session Manager plugin version check not supported$(NC)\n"
	@echo ""
	@printf "$(GREEN)Bootstrap complete. Next: make check-prereqs$(NC)\n"

## check-prereqs: Verify all CLI tools and AWS credentials
check-prereqs:
	@printf "$(GREEN)Checking CLI tools...$(NC)\n"
	@terraform --version | head -1
	@aws --version
	@kubectl version --client 2>/dev/null | head -1
	@helm version --short
	@command -v session-manager-plugin >/dev/null 2>&1 && echo "session-manager-plugin: OK" || \
		printf "$(RED)MISSING: session-manager-plugin. Run: make bootstrap$(NC)\n"
	@echo ""
	@printf "$(GREEN)Checking AWS credentials...$(NC)\n"
	@aws sts get-caller-identity --output table
	@echo ""
	@printf "$(GREEN)All checks passed.$(NC)\n"

# =============================================================================
# Environment Setup (venv + ECR login + kubectl context)
# =============================================================================

## venv: Create virtual environment and install dependencies
venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON3) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV_DIR)/bin/activate

## setup: ECR login, kubectl context, terraform init
setup: venv
	@chmod +x $(SCRIPTS_DIR)/setup.sh
	$(SCRIPTS_DIR)/setup.sh --env $(ENV) --region $(AWS_REGION) --cluster-name $(CLUSTER_NAME)

# =============================================================================
# Infrastructure (Terraform)
# =============================================================================

## infra: Provision AWS infrastructure with Terraform
infra:
	cd $(TF_DIR) && terraform apply -auto-approve -var-file=params/input.tfvars

# =============================================================================
# Kubernetes Storage (EFS)
# =============================================================================

## storage: Deploy EFS StorageClass and PVC
storage:
	envsubst < $(STORAGE_MANIFESTS)/efs-storageclass.yaml | kubectl apply -f -
	kubectl apply -f $(STORAGE_MANIFESTS)/efs-pvc.yaml

## gpu-nodepool: Deploy the GPU Karpenter NodePool for EKS AutoMode
gpu-nodepool:
	kubectl apply -f $(K8S_DIR)/compute/gpu-nodepool.yaml

# =============================================================================
# Container Image (build + push to ECR)
# =============================================================================

## build: Build the training container image
build:
	aws ecr get-login-password --region $(AWS_REGION) | \
		$(RUNTIME) login --username AWS --password-stdin $(DLC_REGISTRY)
	$(RUNTIME) build -t $(IMAGE_NAME):$(IMAGE_TAG) .

## push: Tag and push the container image to ECR
push:
	aws ecr get-login-password --region $(AWS_REGION) | \
		$(RUNTIME) login --username AWS --password-stdin $(ECR_REGISTRY)
	$(RUNTIME) tag $(IMAGE_NAME):$(IMAGE_TAG) $(ECR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	$(RUNTIME) push $(ECR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# =============================================================================
# Training Workload (deploy + run + observe)
# =============================================================================

## run: Render and apply the profiling Job manifest
run:
	envsubst < $(JOB_MANIFEST) | kubectl apply -f -

## logs: Tail logs from the profiling Job
logs:
	kubectl logs -f job/${JOB_NAME}

## status: Check status of training pods
status:
	kubectl get pods -l app=training,method=baseline

# =============================================================================
# SSM Tunnel (private EKS endpoint access)
# =============================================================================

## ssm-tunnel-start: Start SSM tunnel to private EKS API endpoint
ssm-tunnel-start:
	@command -v session-manager-plugin >/dev/null 2>&1 || \
		{ printf "$(RED)Session Manager plugin not installed. Run: make bootstrap$(NC)\n"; exit 1; }
	@chmod +x $(SCRIPTS_DIR)/ssm-kubectl-tunnel.sh
	$(SCRIPTS_DIR)/ssm-kubectl-tunnel.sh start

## ssm-tunnel-stop: Stop the SSM tunnel
ssm-tunnel-stop:
	@chmod +x $(SCRIPTS_DIR)/ssm-kubectl-tunnel.sh
	$(SCRIPTS_DIR)/ssm-kubectl-tunnel.sh stop

## ssm-tunnel-status: Check if SSM tunnel is running
ssm-tunnel-status:
	@chmod +x $(SCRIPTS_DIR)/ssm-kubectl-tunnel.sh
	$(SCRIPTS_DIR)/ssm-kubectl-tunnel.sh status

# =============================================================================
# Cleanup & Teardown
# =============================================================================

## clean: Delete Kubernetes resources
clean:
	kubectl delete job ${JOB_NAME} --ignore-not-found

## clean-venv: Remove the virtual environment
clean-venv:
	rm -rf $(VENV_DIR)

## teardown: Destroy all infrastructure and K8s resources
teardown:
	@chmod +x $(SCRIPTS_DIR)/teardown.sh
	$(SCRIPTS_DIR)/teardown.sh --env $(ENV) --region $(AWS_REGION) --cluster-name $(CLUSTER_NAME)

# =============================================================================
# Full Pipeline
# =============================================================================

## all: Full end-to-end pipeline — setup → infra → storage → build → push → run
all: setup infra build push ssm-tunnel-start storage run