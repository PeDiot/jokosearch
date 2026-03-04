# Joko Search — Generalist vs. Domain-Specific Embeddings for Clothing Product Retrieval

Inspired by [Semantic Search at Scale](https://engineering.hellojoko.com/posts/semantic-search/).

Semantic vector search on the [Joko Clothing Products](https://www.kaggle.com/datasets/b1daeda415bc26ef57c585765a8b1ff504df8bbe460b73b871069d199fea86f5) catalog (17k products). We compare retrieval relevance between two encoders:

- **OpenAI `text-embedding-3-small`** — a generalist text embedding model (1536-dim)
- **[FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP)** — a CLIP model fine-tuned on fashion image–text pairs (512-dim).

Both encoders are evaluated using an LLM-as-a-judge framework inspired by [LLMs Judging LLMs](https://engineering.hellojoko.com/posts/llms-judging-llms-a-new-evaluation-paradigm/) from the Joko Engineering Blog.

## Install

```bash
git clone https://github.com/PeDiot/jokosearch/blob/main/notebook.ipynb
cd jokosearch
uv sync
```

Then create the data directory and download the dataset:

```bash
mkdir data/
```

Download the [Joko Clothing Products dataset](https://www.kaggle.com/datasets/b1daeda415bc26ef57c585765a8b1ff504df8bbe460b73b871069d199fea86f5) from Kaggle and move it to `data/joko_products.csv`.

Set environment variables for the API keys:

```bash
export OPENAI_API_KEY="..."
export HF_API_TOKEN="..."
```

## Workflow

The full pipeline is in `[notebook.ipynb](notebook.ipynb)`.

### 1. Embedding

Each product is encoded twice:

- **OpenAI** encodes `title + description` (concatenated) via the OpenAI API.
- **FashionCLIP** encodes `title` only via a [Hugging Face Space](https://huggingface.co/spaces/precove/fclip_back3) hosting the FashionCLIP encoder.

### 2. Indexing and search

Both sets of embeddings are L2-normalized and loaded into separate in-memory **FAISS** `IndexFlatIP` indices (inner product on normalized vectors = cosine similarity). At query time, the query is embedded with each encoder and the top-k nearest neighbors are retrieved from the corresponding index.

### 3. LLM-as-a-judge evaluation

Two judge strategies are implemented, both backed by `gpt-5-mini` via OpenAI structured outputs:


| Method                       | Description                                                                                                                                                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reference-guided grading** | The judge scores a single query–product pair on a 1–5 relevance scale.                                                                                                                                                  |
| **Pairwise comparison**      | The judge sees both the OpenAI and FashionCLIP results side by side and picks the better match (or declares a tie). This is the primary evaluation method used to reduce API costs (one call per query instead of two). |


### 4. Evaluation queries

Three query categories (30 queries each) were generated with Gemini Pro:


| Category     | Description                                     | Example                                   |
| ------------ | ----------------------------------------------- | ----------------------------------------- |
| [**Simple**](queries/simple.json)  | Basic clothing types with generic colors/styles | `white cotton t-shirt`                    |
| [**Thematic**](queries/thematic.json) | Style, aesthetic, event, or weather context     | `boho chic maxi skirt for music festival` |
| [**Specific**](queries/specific.json) | Exact models, technical materials, brands       | `levis 501 original fit men's jeans`      |


### 5. Results — FashionCLIP win rate

Win rate = share of queries where the pairwise judge preferred the FashionCLIP result over the OpenAI result.


| Category | FashionCLIP wins | OpenAI wins | Ties | FashionCLIP win rate |
| -------- | ---------------- | ----------- | ---- | -------------------- |
| Simple   | —               | —           | —     | —                   |
| Thematic | —                 | —           | —    | —                   |
| Specific | —                | —           | —    | —                    |


## Going further

- Increase `top-k`
- Generate more example queries
- Evaluate with `agent_scoring`