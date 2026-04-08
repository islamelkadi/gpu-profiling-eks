#!/usr/bin/env bash
#
# setup.sh — Initialize the project environment.
#
# Handles:
#   1. ECR login
#   2. kubectl context setup
#   3. Terraform init
#   4. pip install -r requirements.txt (installs Common_Repo)
#
# Usage:
#   ./scripts/setup.sh [--env dev|test|prod] [--region us-west-2] [--cluster-name ai-infra-dev]
#

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

ENV="${ENV:-dev}"
AWS_REGION="${AWS_REGION:-us-west-2}"
CLUSTER_NAME="${CLUSTER_NAME:-ai-infra-${ENV}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${REPO_ROOT}/terraform"

# ─── Parse arguments ────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)          ENV="$2";          CLUSTER_NAME="ai-infra-${ENV}"; shift 2 ;;
    --region)       AWS_REGION="$2";   shift 2 ;;
    --cluster-name) CLUSTER_NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--env dev|test|prod] [--region REGION] [--cluster-name NAME]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo " Environment Setup"
echo "============================================"
echo " Environment:  ${ENV}"
echo " Region:       ${AWS_REGION}"
echo " Cluster:      ${CLUSTER_NAME}"
echo " Repo root:    ${REPO_ROOT}"
echo "============================================"
echo ""

# ─── 1. ECR Login ────────────────────────────────────────────────────────────

echo ">>> Step 1: Logging into Amazon ECR..."

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"

if [[ -z "${AWS_ACCOUNT_ID}" ]]; then
  echo "    WARNING: Could not determine AWS account ID."
  echo "    Make sure your AWS CLI is configured (aws configure) before proceeding."
  echo "    Skipping ECR login."
else
  ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  echo "    ECR registry: ${ECR_REGISTRY}"
  aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${ECR_REGISTRY}" 2>/dev/null \
    && echo "    ECR login successful." \
    || echo "    WARNING: ECR login failed. Docker may not be running."
fi

echo ""

# ─── 2. kubectl Context ─────────────────────────────────────────────────────

echo ">>> Step 2: Configuring kubectl context for EKS cluster..."

if aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws eks update-kubeconfig \
    --name "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --alias "${CLUSTER_NAME}"
  echo "    kubectl context set to '${CLUSTER_NAME}'."
else
  echo "    Cluster '${CLUSTER_NAME}' does not exist yet."
  echo "    Run 'make infra' to provision it first."
  echo "    Then re-run this script to configure kubectl."
fi

echo ""

# ─── 3. Terraform Init ──────────────────────────────────────────────────────

echo ">>> Step 3: Initializing Terraform..."

if [[ -d "${TF_DIR}" ]]; then
  (cd "${TF_DIR}" && terraform init -input=false -upgrade) \
    && echo "    Terraform initialized successfully." \
    || echo "    WARNING: terraform init failed. Check your backend configuration."
else
  echo "    Terraform directory not found at ${TF_DIR}."
fi

echo ""

echo ""
echo "============================================"
echo " Setup complete."
echo " Next step: make help"
echo "============================================"
