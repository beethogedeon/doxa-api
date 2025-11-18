# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
        wget \
        unzip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

RUN wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip

RUN unzip checkpoints_v2_0417.zip -d .

COPY . /app

EXPOSE 3000

CMD ["fastapi", "run", "--port", "3000"]

