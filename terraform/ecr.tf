# -----------------------------------------------------------------------------
# ECR — container image repository for training images
# -----------------------------------------------------------------------------
resource "aws_ecr_repository" "training" {
  name                 = "ai-infra-training"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}
