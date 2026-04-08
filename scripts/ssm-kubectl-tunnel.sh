#!/usr/bin/env bash
# SSM kubectl tunnel — forwards local port 8443 to the EKS private API endpoint
#
# This script:
#   1. Gets the EKS API endpoint hostname and bastion instance ID from Terraform
#   2. Starts an SSM port-forwarding session: localhost:8443 → EKS API:443
#      (DNS resolution happens on the bastion side, inside the VPC)
#   3. Configures kubectl to use localhost:8443
#
# You never shell into the bastion — kubectl runs locally, the bastion is just
# a network relay inside the VPC.
#
# Prerequisites:
#   - AWS CLI v2 with Session Manager plugin (brew install --cask session-manager-plugin)
#   - Terraform state available in terraform/
#   - Bastion instance running
#
# Usage:
#   ./scripts/ssm-kubectl-tunnel.sh start    # Start the tunnel (background)
#   ./scripts/ssm-kubectl-tunnel.sh stop     # Stop the tunnel
#   ./scripts/ssm-kubectl-tunnel.sh status   # Check if tunnel is running

set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
CLUSTER_NAME="${CLUSTER_NAME:-ai-infra-dev}"
LOCAL_PORT=8443
PID_FILE="/tmp/ssm-kubectl-tunnel.pid"
TF_DIR="$(cd "$(dirname "$0")/../terraform" && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

get_instance_id() {
    cd "$TF_DIR" && terraform output -raw bastion_instance_id
}

get_eks_endpoint() {
    cd "$TF_DIR" && terraform output -raw cluster_endpoint
}

start_tunnel() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo -e "${YELLOW}Tunnel already running (PID $(cat "$PID_FILE")). Use 'stop' first.${NC}"
        exit 1
    fi

    # Check Session Manager plugin is installed
    if ! command -v session-manager-plugin &>/dev/null; then
        echo -e "${RED}Session Manager plugin not installed. Run: make bootstrap${NC}"
        exit 1
    fi

    echo -e "${GREEN}Getting bastion instance ID...${NC}"
    local instance_id
    instance_id=$(get_instance_id)
    echo "Instance ID: $instance_id"

    # Check instance is running, start if needed
    local instance_state
    instance_state=$(aws ec2 describe-instance-status \
        --instance-ids "$instance_id" \
        --region "$REGION" \
        --query "InstanceStatuses[0].InstanceState.Name" \
        --output text 2>/dev/null || echo "unknown")

    if [ "$instance_state" != "running" ]; then
        echo -e "${YELLOW}Bastion not running (state: $instance_state). Starting it...${NC}"
        aws ec2 start-instances --instance-ids "$instance_id" --region "$REGION" > /dev/null
        echo "Waiting for instance to be running..."
        aws ec2 wait instance-running --instance-ids "$instance_id" --region "$REGION"
        echo "Waiting for SSM agent to be ready..."
        sleep 30
    fi

    echo -e "${GREEN}Getting EKS endpoint...${NC}"
    local endpoint
    endpoint=$(get_eks_endpoint)
    local endpoint_host
    endpoint_host=$(echo "$endpoint" | sed 's|https://||')
    echo "Endpoint: $endpoint_host"

    echo -e "${GREEN}Starting SSM port-forwarding tunnel...${NC}"
    echo "  localhost:$LOCAL_PORT → $endpoint_host:443"

    aws ssm start-session \
        --target "$instance_id" \
        --document-name AWS-StartPortForwardingSessionToRemoteHost \
        --parameters "{\"host\":[\"$endpoint_host\"],\"portNumber\":[\"443\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}" \
        --region "$REGION" &

    local tunnel_pid=$!
    echo "$tunnel_pid" > "$PID_FILE"
    sleep 3

    if kill -0 "$tunnel_pid" 2>/dev/null; then
        echo -e "${GREEN}Tunnel started (PID $tunnel_pid).${NC}"
        echo ""

        # Update kubeconfig to use the tunnel
        aws eks update-kubeconfig \
            --name "$CLUSTER_NAME" \
            --region "$REGION"

        local account_id
        account_id=$(aws sts get-caller-identity --query Account --output text)
        local cluster_arn="arn:aws:eks:${REGION}:${account_id}:cluster/${CLUSTER_NAME}"

        # Point kubectl at localhost and set TLS server name to the real hostname
        kubectl config set-cluster "$cluster_arn" \
            --server="https://localhost:$LOCAL_PORT" \
            --tls-server-name="$endpoint_host"

        echo ""
        echo -e "${GREEN}kubeconfig updated. kubectl now routes through the SSM tunnel.${NC}"
        echo -e "Test with: ${YELLOW}kubectl get namespaces${NC}"
    else
        echo -e "${RED}Tunnel failed to start. Check SSM plugin installation.${NC}"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop_tunnel() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            sleep 1
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
            echo -e "${GREEN}Tunnel stopped (PID $pid).${NC}"
        else
            echo -e "${YELLOW}Tunnel process $pid already stopped.${NC}"
        fi
        rm -f "$PID_FILE"
    else
        echo -e "${YELLOW}No tunnel PID file found.${NC}"
    fi

    pkill -f "ssm start-session.*${LOCAL_PORT}" 2>/dev/null || true

    local port_pids
    port_pids=$(lsof -ti :"$LOCAL_PORT" 2>/dev/null || true)
    if [ -n "$port_pids" ]; then
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
        echo -e "${YELLOW}Killed orphaned processes on port $LOCAL_PORT.${NC}"
    fi

    # Restore kubeconfig to the original endpoint
    echo -e "${GREEN}Restoring kubeconfig to original EKS endpoint...${NC}"
    aws eks update-kubeconfig \
        --name "$CLUSTER_NAME" \
        --region "$REGION" 2>/dev/null || true
}

tunnel_status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo -e "${GREEN}Tunnel is running (PID $(cat "$PID_FILE")), forwarding localhost:$LOCAL_PORT → EKS API${NC}"
    else
        echo -e "${YELLOW}Tunnel is not running.${NC}"
        rm -f "$PID_FILE" 2>/dev/null
    fi
}

case "${1:-}" in
    start)   start_tunnel ;;
    stop)    stop_tunnel ;;
    status)  tunnel_status ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
