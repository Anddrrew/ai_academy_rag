FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY packages/embedder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/shared/ ./shared/
COPY packages/embedder/src/ .
COPY config.yaml .
COPY .env .

CMD ["fastapi", "run", "main.py", "--port", "8000"]
