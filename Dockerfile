# syntax=docker/dockerfile:1.7
#
# Local, tek konteyner: API + worker + arayüz. Bulut gerektirmez.
#
#   docker compose up                       # http://localhost:8000
#   OZETLE_MODEL=lead docker compose up     # model indirmeden hızlı deneme
#
# Modeller ilk işte indirilir ve "models" biriminde saklanır.

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[api,models]"
COPY frontend ./frontend

RUN useradd --create-home --uid 10001 ozetle \
    && mkdir -p /data /models \
    && chown ozetle /data /models
USER ozetle

ENV OZETLE_FRONTEND_DIR=/app/frontend \
    OZETLE_DATA_DIR=/data \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "-m", "ozetle.service", "--host", "0.0.0.0", "--port", "8000"]
