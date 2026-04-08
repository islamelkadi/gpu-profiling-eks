# -----------------------------------------------------------------------------
# SSM Bastion — kubectl access to the private EKS endpoint
#
# The EKS API is private-only, so kubectl must run from inside the VPC.
# This t3.micro instance sits in a private subnet and is reachable via
# AWS SSM Session Manager — no SSH keys, no public IP.
#
# Usage:
#   aws ssm start-session --target <instance-id>
#   Then run kubectl commands from the session.
# -----------------------------------------------------------------------------

# Latest Amazon Linux 2023 AMI (SSM agent pre-installed)
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

# Security group — egress-only for SSM agent and EKS API
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

# EC2 instance via terraform-aws-modules
module "bastion" {
  source  = "terraform-aws-modules/ec2-instance/aws"
  version = "~> 6.0"

  name          = "${var.cluster_name}-ssm-bastion"
  ami           = data.aws_ami.al2023.id
  instance_type = "t3.micro"

  subnet_id              = module.vpc.private_subnets[0]
  vpc_security_group_ids = [module.bastion_sg.security_group_id]

  create_iam_instance_profile = true
  iam_role_name               = "${var.cluster_name}-ssm-bastion"
  iam_role_policies = {
    ssm = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }

  metadata_options = {
    http_tokens   = "required"
    http_endpoint = "enabled"
  }

  user_data = <<-EOF
    #!/bin/bash
    # Install kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    rm kubectl
  EOF
}
