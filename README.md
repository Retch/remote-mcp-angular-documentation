# Angular Documentation MCP

Semantic search for Angular documentation via MCP protocol.

## Setup

```bash
# 1. Add docs to llms-full/{MAJOR}.txt

# 2. Start services
docker compose up -d

# 3. Import docs
docker compose exec angular-docs-ingestion uv run ingestion.py --version {MAJOR}
```

## Usage

- MCP tool: `search_docs(query, angular_major_version?)` - Search docs (version optional, defaults to configured default)
- Endpoint: `GET /metrics` - Stats

## Config

All configuration is in `config/settings.py`. Add versions and map them to doc files.
