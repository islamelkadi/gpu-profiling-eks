# -----------------------------------------------------------------------------
# VPC — terraform-aws-modules/vpc/aws ~> 5.0
# -----------------------------------------------------------------------------
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
# VPC Endpoints — private connectivity to AWS services
# -----------------------------------------------------------------------------
module "vpc_endpoints" {
  source  = "terraform-aws-modules/vpc/aws//modules/vpc-endpoints"
  version = "~> 5.0"

  vpc_id = module.vpc.vpc_id

  # Interface endpoints (ENI-based, need security group)
  endpoints = {
    eks = {
      service             = "eks"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-eks-endpoint" }
    }
    ecr_api = {
      service             = "ecr.api"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-ecr-api-endpoint" }
    }
    ecr_dkr = {
      service             = "ecr.dkr"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-ecr-dkr-endpoint" }
    }
    sts = {
      service             = "sts"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-sts-endpoint" }
    }
    logs = {
      service             = "logs"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-logs-endpoint" }
    }
    # SSM endpoints (required for bastion in private subnet)
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
    elasticfilesystem = {
      service             = "elasticfilesystem"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [module.vpc_endpoints_sg.security_group_id]
      tags                = { Name = "${var.cluster_name}-efs-endpoint" }
    }
    # Gateway endpoint (no ENI, no cost)
    s3 = {
      service         = "s3"
      service_type    = "Gateway"
      route_table_ids = module.vpc.private_route_table_ids
      tags            = { Name = "${var.cluster_name}-s3-endpoint" }
    }
  }
}

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

# -----------------------------------------------------------------------------
# EKS — terraform-aws-modules/eks/aws ~> 21.0 with Auto Mode
# -----------------------------------------------------------------------------
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.0"

  name               = var.cluster_name
  kubernetes_version = var.cluster_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # EKS Auto Mode
  compute_config = {
    enabled    = true
    node_pools = ["general-purpose"]
  }

  endpoint_public_access                   = false
  endpoint_private_access                  = true
  enable_cluster_creator_admin_permissions = true
}

# Allow bastion to reach the private EKS API endpoint (port 443)
resource "aws_security_group_rule" "bastion_to_eks_api" {
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  description              = "SSM bastion to EKS API"
  security_group_id        = module.eks.cluster_security_group_id
  source_security_group_id = module.bastion_sg.security_group_id
}

# EFS CSI Driver - required for EFS persistent volumes (not included in Auto Mode)
# Needs IRSA role with EFS permissions since Auto Mode nodes don't have instance profiles
resource "aws_eks_addon" "efs_csi" {
  cluster_name             = module.eks.cluster_name
  addon_name               = "aws-efs-csi-driver"
  service_account_role_arn = module.efs_csi_irsa.iam_role_arn
}

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