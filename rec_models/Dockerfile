FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

ARG APT_RETRIES=5
ARG APT_TIMEOUT=30

WORKDIR /app

RUN apt-get update \
        -o Acquire::Retries="${APT_RETRIES}" \
        -o Acquire::http::Timeout="${APT_TIMEOUT}" \
        -o Acquire::https::Timeout="${APT_TIMEOUT}" \
    && apt-get install -y --no-install-recommends \
        -o Acquire::Retries="${APT_RETRIES}" \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/rec_models
RUN ln -s /app/rec_models/data /app/data

EXPOSE 8003

CMD ["uvicorn", "rec_models.app:app", "--host", "0.0.0.0", "--port", "8003"]
