# -----------------------------------------------------------------------------
# EFS — shared storage for training data, checkpoints, and profiler output
# -----------------------------------------------------------------------------
module "efs" {
  source  = "terraform-aws-modules/efs/aws"
  version = "~> 2.0"

  name            = "${var.cluster_name}-training"
  encrypted       = true
  throughput_mode = "elastic"

  # Mount targets in each private subnet
  mount_targets = {
    for idx, subnet_id in module.vpc.private_subnets :
    var.azs[idx] => { subnet_id = subnet_id }
  }

  # Security group — allow NFS from VPC
  security_group_description = "EFS - NFS access from VPC"
  security_group_vpc_id      = module.vpc.vpc_id
  security_group_ingress_rules = {
    vpc = {
      description = "NFS from VPC"
      cidr_ipv4   = var.vpc_cidr
    }
  }

  # Lifecycle — move infrequently accessed files to cheaper storage
  lifecycle_policy = {
    transition_to_ia = "AFTER_30_DAYS"
  }

  enable_backup_policy = true
}
