#!/usr/bin/env bash
#
# teardown.sh — Destroy project resources.
#
# Handles:
#   1. Cleanup of Kubernetes resources
#   2. Terraform destroy for infrastructure
#
# Usage:
#   ./scripts/teardown.sh [--env dev|test|prod] [--region us-west-2] [--cluster-name ai-infra-dev]
#
# WARNING: This will destroy all provisioned infrastructure.
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
echo " Teardown"
echo "============================================"
echo " Environment:  ${ENV}"
echo " Region:       ${AWS_REGION}"
echo " Cluster:      ${CLUSTER_NAME}"
echo "============================================"
echo ""
read -rp "This will DESTROY all infrastructure for env '${ENV}'. Continue? [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

echo ""

# ─── 1. Cleanup Kubernetes Resources ────────────────────────────────────────

echo ">>> Step 1: Cleaning up Kubernetes resources..."

if kubectl cluster-info >/dev/null 2>&1; then
  echo "    Deleting training Jobs..."
  kubectl delete jobs -l app=training --ignore-not-found=true 2>/dev/null || true

  echo "    Kubernetes resources cleaned up."
else
  echo "    kubectl cannot reach the cluster. Skipping K8s cleanup."
fi

echo ""

# ─── 2. Terraform Destroy ───────────────────────────────────────────────────

echo ">>> Step 2: Destroying Terraform-managed infrastructure..."

if [[ -d "${TF_DIR}" ]] && [[ -f "${TF_DIR}/main.tf" ]]; then
  TFVARS_FILE="${TF_DIR}/params/input.tfvars"

  if [[ -f "${TFVARS_FILE}" ]]; then
    (cd "${TF_DIR}" && terraform destroy -var-file="params/input.tfvars" -auto-approve) \
      && echo "    Terraform destroy completed." \
      || echo "    WARNING: Terraform destroy encountered errors."
  else
    echo "    No tfvars file found at ${TFVARS_FILE}."
    (cd "${TF_DIR}" && terraform destroy -auto-approve) \
      || echo "    WARNING: Terraform destroy failed."
  fi
else
  echo "    No Terraform configuration found. Nothing to destroy."
fi

echo ""
echo "============================================"
echo " Teardown complete."
echo "============================================"
