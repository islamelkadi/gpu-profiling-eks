# =============================================================================
# GPU Profiling Infrastructure on AWS
# =============================================================================
# This Terraform configuration provisions:
# 1. VPC with private/public subnets and NAT Gateway
# 2. VPC Endpoints for private AWS service connectivity
# 3. EKS Auto Mode cluster with GPU support
# 4. EFS filesystem for persistent training data
# 5. ECR repository for container images
# 6. SSM bastion for secure kubectl access
# =============================================================================

# -----------------------------------------------------------------------------
# 1. VPC — Network Foundation
# -----------------------------------------------------------------------------
# Creates isolated network with private subnets for GPU nodes and public
# subnets for NAT Gateway. Private subnets tagged for EKS cluster discovery.
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = var.azs
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "prod"
  enable_dns_hostnames = true
  enable_dns_support   = true

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# -----------------------------------------------------------------------------
# 2. VPC Endpoints — Private AWS Service Connectivity
# -----------------------------------------------------------------------------
# Eliminates NAT Gateway costs for AWS API calls and improves security by
# keeping traffic within AWS backbone. Essential for private EKS clusters.

# Security group for interface VPC endpoints
module "vpc_endpoints_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"

  name        = "${var.cluster_name}-vpce"
  description = "Security group for VPC endpoints"
  vpc_id      = module.vpc.vpc_id

  ingress_with_cidr_blocks = [
    {
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      description = "HTTPS from VPC"
      cidr_blocks = var.vpc_cidr
    }
  ]
}

# VPC Endpoints for AWS services
module "vpc_endpoints" {
  source  = "terraform-aws-modules/vpc/aws//modules/vpc-endpoints"
  version = "~> 5.0"

  vpc_id = module.vpc.vpc_id

  # Interface endpoints (ENI-based, need security group)
  endpoints = {
    # EKS cluster API endpoint
    eks = {
      service             = "eks"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-eks-endpoint" }
    }
    # ECR API for container image metadata
    ecr_api = {
      service             = "ecr.api"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-ecr-api-endpoint" }
    }
    # ECR Docker registry for image layers
    ecr_dkr = {
      service             = "ecr.dkr"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-ecr-dkr-endpoint" }
    }
    # STS for IAM token exchange (IRSA)
    sts = {
      service             = "sts"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-sts-endpoint" }
    }
    # CloudWatch Logs for container logging
    logs = {
      service             = "logs"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-logs-endpoint" }
    }
    # SSM endpoints for bastion host connectivity
    ssm = {
      service             = "ssm"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-ssm-endpoint" }
    }
    ssmmessages = {
      service             = "ssmmessages"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-ssmmessages-endpoint" }
    }
    ec2messages = {
      service             = "ec2messages"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-ec2messages-endpoint" }
    }
    # EFS endpoint for persistent storage
    elasticfilesystem = {
      service             = "elasticfilesystem"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-efs-endpoint" }
    }
    # S3 Gateway endpoint (no ENI, no cost)
    s3 = {
      service         = "s3"
      service_type    = "Gateway"
      route_table_ids = module.vpc.private_route_table_ids
      tags            = { Name = "${var.cluster_name}-s3-endpoint" }
    }
  }
}

# -----------------------------------------------------------------------------
# 3. EKS Auto Mode Cluster — Managed Kubernetes with GPU Support
# -----------------------------------------------------------------------------
# Auto Mode handles all node management, including GPU instance provisioning
# via Karpenter. Private API endpoint requires bastion for kubectl access.
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.0"

  name               = var.cluster_name
  kubernetes_version = var.cluster_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # EKS Auto Mode configuration
  compute_config = {
    enabled    = true
    node_pools = ["general-purpose"]
  }

  # Private cluster for security
  endpoint_public_access                   = false
  endpoint_private_access                  = true
  enable_cluster_creator_admin_permissions = true
}

# Allow bastion to reach the private EKS API endpoint
resource "aws_security_group_rule" "bastion_to_eks_api" {
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  description              = "SSM bastion to EKS API"
  security_group_id        = module.eks.cluster_security_group_id
  source_security_group_id = module.bastion_sg.security_group_id
}

# EFS CSI Driver addon for persistent volumes
# Auto Mode doesn't include this by default, but we need it for training data
resource "aws_eks_addon" "efs_csi" {
  cluster_name             = module.eks.cluster_name
  addon_name               = "aws-efs-csi-driver"
  service_account_role_arn = module.efs_csi_irsa.iam_role_arn
}

# IRSA role for EFS CSI driver
module "efs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.cluster_name}-efs-csi-driver"
  attach_efs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:efs-csi-controller-sa"]
    }
  }
}

# -----------------------------------------------------------------------------
# 4. EFS — Shared Storage for Training Data and Outputs
# -----------------------------------------------------------------------------
# Persistent filesystem for datasets, checkpoints, and profiler traces.
# Elastic throughput scales automatically with workload demands.
module "efs" {
  source  = "terraform-aws-modules/efs/aws"
  version = "~> 2.0"

  name            = "${var.cluster_name}-training"
  encrypted       = true
  throughput_mode = "elastic"

  # Mount targets in each private subnet for high availability
  mount_targets = {
    for idx, subnet_id in module.vpc.private_subnets :
    var.azs[idx] => { subnet_id = subnet_id }
  }

  # Security group for NFS access from VPC
  security_group_description = "EFS - NFS access from VPC"
  security_group_vpc_id      = module.vpc.vpc_id
  security_group_ingress_rules = {
    vpc = {
      description = "NFS from VPC"
      cidr_ipv4   = var.vpc_cidr
    }
  }

  # Lifecycle policy for cost optimization
  lifecycle_policy = {
    transition_to_ia = "AFTER_30_DAYS"
  }

  enable_backup_policy = true
}

# -----------------------------------------------------------------------------
# 5. ECR — Container Image Repository
# -----------------------------------------------------------------------------
# Private registry for training container images with vulnerability scanning.
resource "aws_ecr_repository" "training" {
  name                 = "ai-infra-training"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# -----------------------------------------------------------------------------
# 6. SSM Bastion — Secure kubectl Access
# -----------------------------------------------------------------------------
# t3.micro instance in private subnet for kubectl access to private EKS API.
# No SSH keys or public IP - access via AWS SSM Session Manager only.

# Latest Amazon Linux 2023 AMI with SSM agent pre-installed
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security group for bastion - egress-only for SSM and EKS API
module "bastion_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"

  name        = "${var.cluster_name}-ssm-bastion"
  description = "SSM bastion - outbound HTTPS for SSM agent and EKS API"
  vpc_id      = module.vpc.vpc_id

  egress_with_cidr_blocks = [
    {
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      description = "HTTPS for SSM agent + EKS API"
      cidr_blocks = "0.0.0.0/0"
    }
  ]
}

# Bastion EC2 instance with kubectl pre-installed
module "bastion" {
  source  = "terraform-aws-modules/ec2-instance/aws"
  version = "~> 6.0"

  name          = "${var.cluster_name}-ssm-bastion"
  ami           = data.aws_ami.al2023.id
  instance_type = "t3.micro"

  subnet_id              = module.vpc.private_subnets[0]
  vpc_security_group_ids = [module.bastion_sg.security_group_id]

  # IAM role for SSM access
  create_iam_instance_profile = true
  iam_role_name               = "${var.cluster_name}-ssm-bastion"
  iam_role_policies = {
    ssm = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }

  # Security hardening
  metadata_options = {
    http_tokens   = "required"
    http_endpoint = "enabled"
  }

  # Install kubectl on boot
  user_data = <<-EOF
    #!/bin/bash
    # Install kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    rm kubectl
  EOF
}