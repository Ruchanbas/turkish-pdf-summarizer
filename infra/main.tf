# ozetle — AWS serverless yığını.
#
#   tarayıcı ──▶ CloudFront ──┬─▶ S3 (arayüz)
#                             └─▶ /api/* ─▶ API Gateway ─▶ Lambda (API, zip)
#   tarayıcı ──ön imzalı POST──▶ S3 (yüklemeler) ──olay──▶ SQS ─▶ Lambda (worker, imaj)
#                                                              └─▶ DLQ ─▶ alarm
#   iş durumu, önbellek, kota: DynamoDB (TTL)
#
# Boştayken maliyet ~0: her şey kullanım başına ücretlendirilir. Tek sabit
# kalem ECR'deki imaj deposudur (birkaç GB, ayda kuruşlar).
#
# Durum S3'te tutulur; kova infra/bootstrap ile oluşturulur:
#   terraform init -backend-config="bucket=<state_bucket>" -backend-config="region=eu-central-1"

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
  backend "s3" {
    key          = "ozetle/app.tfstate"
    use_lockfile = true
    encrypt      = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "ozetle", ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account   = data.aws_caller_identity.current.account_id
  partition = data.aws_partition.current.partition
  prefix    = "uploads/"

  common_env = {
    OZETLE_BACKEND         = "aws"
    OZETLE_TABLE_NAME      = aws_dynamodb_table.jobs.name
    OZETLE_UPLOAD_BUCKET   = aws_s3_bucket.uploads.bucket
    OZETLE_UPLOAD_PREFIX   = local.prefix
    OZETLE_MODEL           = var.model
    OZETLE_DAILY_JOB_LIMIT = tostring(var.daily_job_limit)
    OZETLE_LOG_LEVEL       = "INFO"
  }
}
