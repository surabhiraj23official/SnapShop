# Shopee Product Matching Web Application

FastAPI-based search interface powered by a 5-model ensemble for multimodal
product matching.

## Architecture

The inference engine uses three search modalities:

| Modality     | Models                                            | FAISS Index      |
| ------------ | ------------------------------------------------- | ---------------- |
| **Image**    | DeiT Small + EfficientNet-B3                      | `img_index`      |
| **Text**     | BERT Indonesian + BERT Multilingual + XLM-RoBERTa | `bert_index`     |
| **Combined** | All 5 models concatenated                         | `combined_index` |

## Shared Project Layout

The app reads from these shared directories:

- `data/shopee-product-matching/` - dataset CSVs and product images
- `artifacts/checkpoints/` - trained `.pth` encoder checkpoints
- `artifacts/cache/` - precomputed embeddings and `lgbm_ranker.txt`

Expected checkpoint files:

- `deit_small.pth`
- `efficientnet_b3.pth`
- `bert_indonesian.pth`
- `bert_multilingual.pth`
- `xlm_roberta.pth`

Current cache files:

- `img_feats.npy` - image feature vectors
- `text_feats.npy` - text feature vectors
- `combined_feats.npy` - concatenated `[text, image]` vectors

## API

### `POST /api/search`

Multipart form with optional `image` file and/or `text` field.

Returns:

```json
{
  "results": [
    {
      "title": "Product name",
      "image": "filename.jpg",
      "score": 0.92,
      "price": "$29.99",
      "posting_id": "abc123"
    }
  ]
}
```

## Running

```bash
cd webapp/app
pip install -r requirements.txt
python run.py
```

Server starts at `http://localhost:8000`.
