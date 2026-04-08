output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "cluster_endpoint" {
  description = "EKS cluster API server endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority" {
  description = "EKS cluster certificate authority data"
  value       = module.eks.cluster_certificate_authority_data
}

output "oidc_provider_arn" {
  description = "ARN of the OIDC provider for IRSA"
  value       = module.eks.oidc_provider_arn
}

output "bastion_instance_id" {
  description = "Instance ID of the SSM bastion host"
  value       = module.bastion.id
}

output "efs_id" {
  description = "ID of the EFS file system for training data"
  value       = module.efs.id
}

output "efs_dns_name" {
  description = "DNS name of the EFS file system"
  value       = module.efs.dns_name
}

output "ecr_repository_url" {
  description = "URL of the ECR repository for training images"
  value       = aws_ecr_repository.training.repository_url
}
