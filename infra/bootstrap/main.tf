# Tek seferlik kurulum — yönetici kimliğiyle, yerelde bir kez çalıştırılır:
#
#   cd infra/bootstrap
#   terraform init && terraform apply
#
# Oluşturduğu şeyler:
#   - Terraform durum kovası (S3, sürümlü, şifreli, yalnızca TLS)
#   - GitHub Actions OIDC sağlayıcısı (hesapta yoksa)
#   - Bu depo için deploy rolü: yalnızca main dalından üstlenilebilir,
#     yetkileri "ozetle-*" adlı kaynaklarla sınırlı
#
# Sonrasında hiçbir statik AWS anahtarı GitHub'a yazılmaz.

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "ozetle", ManagedBy = "terraform-bootstrap" }
  }
}

variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "github_repo" {
  description = "sahip/depo"
  type        = string
  default     = "Ruchanbas/turkish-pdf-summarizer"
}

variable "create_oidc_provider" {
  description = <<-EOT
    Hesapta GitHub OIDC sağlayıcısı yoksa true. Bir hesapta bu URL için
    yalnızca bir sağlayıcı olabilir; başka bir proje (ör. rg-ilan-botu)
    zaten oluşturduysa false bırakın. Kontrol:
      aws iam list-open-id-connect-providers
  EOT
  type        = bool
  default     = false
}

variable "name" {
  type    = string
  default = "ozetle"
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account   = data.aws_caller_identity.current.account_id
  partition = data.aws_partition.current.partition
  owner     = split("/", var.github_repo)[0]
  repo      = split("/", var.github_repo)[1]
  oidc_arn = (
    var.create_oidc_provider
    ? aws_iam_openid_connect_provider.github[0].arn
    : data.aws_iam_openid_connect_provider.github[0].arn
  )
}

# ---------------------------------------------------------------------------
# Durum kovası
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "state" {
  bucket = "${var.name}-tfstate-${local.account}"
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "state" {
  bucket = aws_s3_bucket.state.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "OnlyTLS"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

# ---------------------------------------------------------------------------
# GitHub OIDC
# ---------------------------------------------------------------------------

resource "aws_iam_openid_connect_provider" "github" {
  count          = var.create_oidc_provider ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_oidc_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_role" "deploy" {
  name                 = "${var.name}-github-deploy"
  max_session_duration = 3600
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        # Yalnızca main dalı. 15 Temmuz 2026'dan sonra açılan depolarda
        # "sub" sahip ve depo adının yanına değişmez kimlikler ekler
        # (repo:sahip@<id>/depo@<id>:ref:...); eski biçim de kabul edilir.
        StringLike = {
          "token.actions.githubusercontent.com:sub" = [
            "repo:${local.owner}@*/${local.repo}@*:ref:refs/heads/main",
            "repo:${var.github_repo}:ref:refs/heads/main",
          ]
        }
      }
    }]
  })
}

# Deploy rolünün yetkileri: yalnızca bu projenin kaynakları.
data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "State"
    actions   = ["s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]
  }

  statement {
    sid     = "ProjectBuckets"
    actions = ["s3:*"]
    resources = [
      "arn:${local.partition}:s3:::${var.name}-*-${local.account}",
      "arn:${local.partition}:s3:::${var.name}-*-${local.account}/*",
    ]
  }

  statement {
    sid       = "Functions"
    actions   = ["lambda:*"]
    resources = ["arn:${local.partition}:lambda:${var.region}:${local.account}:function:${var.name}-*"]
  }

  statement {
    sid = "EventSourceMappings"
    actions = [
      "lambda:CreateEventSourceMapping", "lambda:GetEventSourceMapping",
      "lambda:UpdateEventSourceMapping", "lambda:DeleteEventSourceMapping",
      "lambda:ListEventSourceMappings", "lambda:TagResource", "lambda:ListTags",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "Table"
    actions   = ["dynamodb:*"]
    resources = ["arn:${local.partition}:dynamodb:${var.region}:${local.account}:table/${var.name}-*"]
  }

  statement {
    sid       = "Queues"
    actions   = ["sqs:*"]
    resources = ["arn:${local.partition}:sqs:${var.region}:${local.account}:${var.name}-*"]
  }

  statement {
    sid       = "Registry"
    actions   = ["ecr:*"]
    resources = ["arn:${local.partition}:ecr:${var.region}:${local.account}:repository/${var.name}-*"]
  }

  statement {
    sid       = "RegistryLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "Roles"
    actions = [
      "iam:CreateRole", "iam:DeleteRole", "iam:GetRole", "iam:UpdateRole", "iam:TagRole",
      "iam:UntagRole", "iam:PutRolePolicy", "iam:GetRolePolicy", "iam:DeleteRolePolicy",
      "iam:ListRolePolicies", "iam:ListAttachedRolePolicies", "iam:ListInstanceProfilesForRole",
      "iam:UpdateAssumeRolePolicy",
    ]
    resources = ["arn:${local.partition}:iam::${local.account}:role/${var.name}-*"]
  }

  statement {
    sid       = "PassRolesToLambda"
    actions   = ["iam:PassRole"]
    resources = ["arn:${local.partition}:iam::${local.account}:role/${var.name}-*"]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["lambda.amazonaws.com"]
    }
  }

  statement {
    sid = "LogsAndAlarms"
    actions = [
      "logs:*", "cloudwatch:PutMetricAlarm", "cloudwatch:DeleteAlarms",
      "cloudwatch:DescribeAlarms", "cloudwatch:ListTagsForResource", "cloudwatch:TagResource",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "Alerts"
    actions   = ["sns:*"]
    resources = ["arn:${local.partition}:sns:${var.region}:${local.account}:${var.name}-*"]
  }

  statement {
    sid       = "Budget"
    actions   = ["budgets:*"]
    resources = ["arn:${local.partition}:budgets::${local.account}:budget/${var.name}-*"]
  }

  statement {
    sid       = "ServiceLinkedRoles"
    actions   = ["iam:CreateServiceLinkedRole"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "iam:AWSServiceName"
      values   = ["ops.apigateway.amazonaws.com", "lambda.amazonaws.com"]
    }
  }

  # API Gateway ve CloudFront kaynak düzeyinde ad kısıtını desteklemez.
  statement {
    sid       = "EdgeAndApi"
    actions   = ["apigateway:*", "cloudfront:*"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "${var.name}-deploy"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}

output "deploy_role_arn" {
  description = "GitHub → Settings → Variables: AWS_ROLE_ARN"
  value       = aws_iam_role.deploy.arn
}

output "state_bucket" {
  description = "GitHub → Settings → Variables: TF_STATE_BUCKET"
  value       = aws_s3_bucket.state.bucket
}
