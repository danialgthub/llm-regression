FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ libffi-dev libssl-dev libxml2-dev libxslt-dev zlib1g-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure Python can find /app/src
ENV PYTHONPATH=/app

CMD ["python", "src/eval_runner.py"]
