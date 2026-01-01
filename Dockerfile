FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    libboost-all-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv インストール
RUN pip install uv

# 依存関係
COPY pyproject.toml .
RUN uv pip install --system -e .

# LEANN
COPY leann/ ./leann/
RUN cd leann && uv pip install --system -e .

# アプリケーション
COPY src/ ./src/

# データディレクトリ
RUN mkdir -p /app/data/indexes /app/data/uploads

# 埋め込みモデル事前ダウンロード
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('cl-nagoya/ruri-v3-310m')"

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
