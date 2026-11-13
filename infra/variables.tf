variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "name" {
  description = "Tüm kaynak adlarının öneki (bootstrap'teki deploy rolü bu öneke yetkilidir)"
  type        = string
  default     = "ozetle"
}

variable "worker_image_tag" {
  description = "ECR'deki worker imajının etiketi (deploy workflow'u commit SHA'sını verir)"
  type        = string
}

variable "api_zip_path" {
  description = "scripts/build_api_zip.sh çıktısı"
  type        = string
  default     = "../dist/api.zip"
}

variable "model" {
  description = "İmajın içine gömülen özetleme modeli (katalog anahtarı); imaj derlemesiyle aynı olmalı"
  type        = string
  default     = "mt5-small"
}

variable "worker_memory_mb" {
  description = "Worker belleği. Lambda'da CPU bellekle orantılıdır; yeni hesaplarda üst sınır 3008 olabilir."
  type        = number
  default     = 3008
}

variable "worker_timeout_seconds" {
  type    = number
  default = 300
}

variable "worker_max_concurrency" {
  description = "Aynı anda çalışan en fazla worker (SQS ölçekleme sınırı, en az 2)"
  type        = number
  default     = 2
}

variable "daily_job_limit" {
  description = "Günlük yeni iş sınırı — herkese açık demoda maliyet freni"
  type        = number
  default     = 300
}

variable "alert_email" {
  description = "Alarm ve bütçe bildirimleri. Boşsa e-posta aboneliği kurulmaz."
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  type    = number
  default = 5
}

variable "log_retention_days" {
  type    = number
  default = 14
}
