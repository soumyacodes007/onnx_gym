FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1     UV_LINK_MODE=copy     PIP_NO_CACHE_DIR=1     ENABLE_WEB_INTERFACE=true     PATH="/root/.local/bin:${PATH}"     PYTHONPATH="/app/env"

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends     curl     libgomp1     && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock openenv.yaml README.md __init__.py client.py models.py tasks.py inference.py requirements.txt ./
COPY server ./server
COPY tests ./tests

RUN uv sync --frozen --no-dev

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3   CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uv run uvicorn server.app:app --host 0.0.0.0 --port 7860"]
