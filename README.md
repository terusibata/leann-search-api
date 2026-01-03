# LEANN Search API

LEANNベースの軽量・高精度なベクトル検索APIサーバー。外部AIバックエンドのRAG検索ツールとして利用できます。

## 機能

- **インデックス管理**: 複数インデックスの作成・削除・一覧
- **ドキュメント管理**: メタデータ付きドキュメントのCRUD操作
- **検索**: セマンティック検索 + メタデータフィルタリング + Grep検索 + ハイブリッド検索

## 技術スタック

| 項目 | 技術 |
|------|------|
| 言語 | Python 3.11+ |
| フレームワーク | FastAPI |
| 検索エンジン | LEANN (pip install leann) |
| 埋め込みモデル | cl-nagoya/ruri-v3-310m (デフォルト) |
| コンテナ | Docker + Docker Compose |
| パッケージ管理 | uv |

## セットアップ

### 前提条件

- Python 3.11+
- uv (パッケージマネージャー)
- Docker & Docker Compose (オプション)

### 方法1: Docker Compose (推奨)

```bash
# リポジトリをクローン
git clone <repository-url>
cd leann-search-api

# 環境変数を設定
cp .env.example .env

# Docker起動
docker-compose up -d

# ヘルスチェック
curl http://localhost:8000/health
```

### 方法2: ローカル開発

```bash
# 仮想環境を作成
uv venv
source .venv/bin/activate

# 依存関係をインストール (LEANNを含む)
uv pip install -e ".[dev]"

# 開発サーバーを起動
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## LEANN ライブラリ

本プロジェクトは [LEANN](https://pypi.org/project/leann/) ベクトル検索ライブラリを使用しています。

```bash
pip install leann
```

### 主要コンポーネント

- **LeannBuilder**: インデックス構築
  - `add_text(text)`: テキストを追加
  - `build_index(path)`: インデックスを構築・保存

- **LeannSearcher**: 検索実行
  - `search(query, top_k)`: 類似検索を実行

## Ollama統合（プロキシ対応）

このプロジェクトは[Ollama](https://ollama.com/)と統合して、ローカルで埋め込みモデルを実行できます。プロキシ環境下でも動作します。

### メリット

- **完全プライベート**: データが外部に送信されない
- **オフライン動作**: モデルダウンロード後はインターネット不要
- **無料**: サブスクリプション費用なし
- **プロキシ対応**: 企業環境でも使用可能

### セットアップ手順

#### 1. Ollamaのインストール（プロキシ環境）

**Dockerを使用する場合（推奨）:**

```bash
docker run -d \
  -e HTTPS_PROXY=https://your.proxy.server:port \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama
```

**Linuxにインストールする場合:**

```bash
# インストール
curl -fsSL https://ollama.com/install.sh | sh

# プロキシ設定（systemd）
sudo systemctl edit ollama.service
```

以下を追加:
```ini
[Service]
Environment="HTTPS_PROXY=https://your.proxy.server:port"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**注意**: `HTTP_PROXY`は設定しないでください（Ollamaの接続に問題が発生する可能性があります）

#### 2. 埋め込みモデルのダウンロード

```bash
# 推奨: mxbai-embed-large (334M パラメータ)
ollama pull mxbai-embed-large

# または軽量版
ollama pull nomic-embed-text  # 137M パラメータ
ollama pull all-minilm        # 23M パラメータ

# 確認
curl http://localhost:11434/api/tags
```

#### 3. 環境変数の設定

`.env`ファイルを編集:

```bash
# Ollama使用時の設定
EMBEDDING_MODEL=mxbai-embed-large
EMBEDDING_MODE=ollama

# その他の設定
LEANN_BACKEND=hnsw
GRAPH_DEGREE=32
BUILD_COMPLEXITY=64
```

#### 4. サーバー起動

```bash
# Docker Composeの場合
docker-compose up -d

# ローカル開発の場合
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 対応埋め込みモデル

| モデル | パラメータ数 | 用途 |
|--------|-------------|------|
| `mxbai-embed-large` | 334M | 高精度（推奨） |
| `nomic-embed-text` | 137M | バランス型 |
| `all-minilm` | 23M | 軽量・高速 |

詳細は[Ollama Embedding Models](https://ollama.com/blog/embedding-models)を参照

## API仕様

### ベースURL

```
http://localhost:8000/api/v1
```

### エンドポイント一覧

#### ヘルスチェック

```bash
GET /health
```

#### インデックス管理

```bash
GET    /api/v1/indexes                    # インデックス一覧
POST   /api/v1/indexes                    # インデックス作成
GET    /api/v1/indexes/{name}             # インデックス詳細
DELETE /api/v1/indexes/{name}             # インデックス削除
POST   /api/v1/indexes/{name}/rebuild     # インデックス再構築
```

#### ドキュメント管理

```bash
POST   /api/v1/indexes/{name}/documents              # ドキュメント追加
POST   /api/v1/indexes/{name}/documents/file         # ファイルアップロード
GET    /api/v1/indexes/{name}/documents              # ドキュメント一覧
GET    /api/v1/indexes/{name}/documents/{id}         # ドキュメント詳細
PUT    /api/v1/indexes/{name}/documents/{id}         # ドキュメント更新
PATCH  /api/v1/indexes/{name}/documents/{id}/metadata # メタデータ更新
DELETE /api/v1/indexes/{name}/documents/{id}         # ドキュメント削除
POST   /api/v1/indexes/{name}/documents/bulk-delete  # 一括削除
```

#### 検索

```bash
POST /api/v1/indexes/{name}/search         # セマンティック検索
POST /api/v1/indexes/{name}/search/grep    # キーワード検索
POST /api/v1/indexes/{name}/search/hybrid  # ハイブリッド検索
POST /api/v1/indexes/{name}/search/batch   # 複数クエリ一括検索
```

## 使用例

### インデックス作成

```bash
curl -X POST http://localhost:8000/api/v1/indexes \
  -H "Content-Type: application/json" \
  -d '{"name": "company_docs"}'
```

### ドキュメント追加

```bash
curl -X POST http://localhost:8000/api/v1/indexes/company_docs/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "id": "doc_001",
      "content": "経費精算の申請期限は経費発生日から1ヶ月以内です。",
      "metadata": {"category": "manual", "department": "経理"}
    }]
  }'
```

### インデックス再構築

ドキュメント追加後、検索を有効にするためにインデックスを再構築します:

```bash
curl -X POST http://localhost:8000/api/v1/indexes/company_docs/rebuild
```

### 検索

```bash
curl -X POST http://localhost:8000/api/v1/indexes/company_docs/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "経費精算の期限",
    "top_k": 5,
    "metadata_filters": {"category": {"==": "manual"}}
  }'
```

## 外部AIバックエンドとの連携

### Python

```python
import httpx

SEARCH_API = "http://leann-search-api:8000/api/v1"

async def search_documents(query: str, index: str, top_k: int = 5):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SEARCH_API}/indexes/{index}/search",
            json={"query": query, "top_k": top_k}
        )
        return response.json()["data"]["results"]

async def build_rag_context(question: str):
    results = await search_documents(question, "company_docs", top_k=5)
    return "\n\n".join([r["content"] for r in results])
```

### TypeScript

```typescript
const SEARCH_API = "http://leann-search-api:8000/api/v1";

async function searchDocuments(query: string, index: string, topK = 5) {
  const response = await fetch(`${SEARCH_API}/indexes/${index}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK })
  });
  const data = await response.json();
  return data.data.results;
}
```

## テスト

```bash
# テストを実行
pytest

# カバレッジ付き
pytest --cov=src
```

## 環境変数

### サーバー設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `HOST` | `0.0.0.0` | サーバーホスト |
| `PORT` | `8000` | サーバーポート |
| `WORKERS` | `4` | ワーカープロセス数 |
| `LOG_LEVEL` | `info` | ログレベル |

### LEANN設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `INDEX_DIR` | `./data/indexes` | インデックス保存先ディレクトリ |
| `LEANN_BACKEND` | `hnsw` | バックエンドアルゴリズム（`hnsw` または `diskann`） |
| `GRAPH_DEGREE` | `32` | グラフ次数（高いほど精度向上、メモリ増加） |
| `BUILD_COMPLEXITY` | `64` | 構築時の複雑度（高いほど精度向上、構築時間増加） |
| `SEARCH_COMPLEXITY` | `32` | 検索時の複雑度（高いほど精度向上、検索時間増加） |

### 埋め込みモデル設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `EMBEDDING_MODEL` | `cl-nagoya/ruri-v3-310m` | 埋め込みモデル名 |
| `EMBEDDING_MODE` | `sentence-transformers` | 埋め込みモード（`sentence-transformers`, `ollama`, `openai`, `mlx`） |

**埋め込みモード詳細:**
- `sentence-transformers`: Hugging Faceモデル（オフライン動作）
- `ollama`: Ollamaローカルモデル（プロキシ対応、完全プライベート）
- `openai`: OpenAI API（要APIキー）
- `mlx`: Apple Silicon最適化モデル

### ドキュメント設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `DEFAULT_CHUNK_SIZE` | `512` | デフォルトチャンクサイズ（文字数） |
| `DEFAULT_CHUNK_OVERLAP` | `64` | チャンク間のオーバーラップ（文字数） |
| `MAX_UPLOAD_SIZE_MB` | `10` | 最大アップロードサイズ（MB） |

### 検索設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `DEFAULT_TOP_K` | `10` | デフォルト検索件数 |
| `MAX_TOP_K` | `100` | 最大検索件数 |

## ライセンス

MIT
