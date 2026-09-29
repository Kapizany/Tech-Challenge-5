FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[cloud]'
COPY scripts ./scripts
COPY configs ./configs

EXPOSE 8080
CMD ["sh", "-c", "uvicorn adaptive_offers.api:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
