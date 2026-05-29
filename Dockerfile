# Builder: install deps into /install prefix, kept out of final image
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Runner: minimal runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/cache \
    && chown appuser:appuser /app/cache

COPY dashboard.py .
COPY data_provider/ data_provider/
COPY models/ models/
COPY .streamlit/ .streamlit/

ENV CACHE_DIR=/app/cache

EXPOSE 8501

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard.py"]
