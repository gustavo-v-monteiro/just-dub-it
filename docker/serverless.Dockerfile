FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    LTX_MODEL_ROOT=/models \
    LTX_SCRATCH_ROOT=/tmp/jobs \
    LTX_PROVIDER=runpod

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages ./packages

RUN uv sync --frozen --package ltx-service --no-dev

CMD ["python", "-m", "ltx_service.providers.runpod_handler"]
