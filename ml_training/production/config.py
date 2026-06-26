"""
Shopee Product Matching production training configuration.

Stores shared datasets and generated artifacts in top-level project
directories so training and serving read from the same locations.
"""

import os
from pathlib import Path


class Config:
    # -- Directory Structure -------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent.parent

    DATA_DIR = PROJECT_ROOT / 'data' / 'shopee-product-matching'
    ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
    OUTPUT_DIR = ARTIFACTS_DIR
    CHECKPOINT_DIR = ARTIFACTS_DIR / 'checkpoints'
    CACHE_DIR = ARTIFACTS_DIR / 'cache'

    TRAIN_CSV = DATA_DIR / 'train.csv'
    TRAIN_IMAGES = DATA_DIR / 'train_images'
    IMG_FEATURES_PATH = CACHE_DIR / 'img_feats.npy'
    TEXT_FEATURES_PATH = CACHE_DIR / 'text_feats.npy'
    COMBINED_FEATURES_PATH = CACHE_DIR / 'combined_feats.npy'
    LGBM_MODEL_PATH = CACHE_DIR / 'lgbm_ranker.txt'

    SEED = 42
    NUM_WORKERS = 2
    DEVICE = 'cuda'

    # -- 1st Stage: Metric Learning Models -----------------------------------

    # Sample weighting for micro-F1 optimization
    # Smaller label groups get higher weight so the model does not
    # ignore minority groups.  weight = 1 / (group_size ** 0.4)
    SAMPLE_WEIGHT_POWER = 0.4

    IMAGE_MODELS = {
        'deit_small': {
            'backbone': 'vit_deit_small_distilled_patch16_224',
            'img_size': 224,
            'test_size': 224,
            'fc_dim': 512,
            'batch_size': 16,
            'epochs': 15,
            'lr': 3e-4,
            's': 30,
            'margin': 0.5,
            'p': 3,
            'loss_fn': 'curricularface',       # better than ArcFace for images
            'optimizer': 'sam',                 # finds flat minima
        },
        'efficientnet_b3': {
            'backbone': 'tf_efficientnet_b3_ns',
            'img_size': 512,
            'test_size': 384,
            'fc_dim': 512,
            'batch_size': 8,
            'epochs': 15,
            'lr': 3e-4,
            's': 30,
            'margin': 0.5,
            'p': 3,
            'p_eval': 4,                        # higher GeM power at inference
            'loss_fn': 'curricularface',
            'optimizer': 'sam',
        },
    }

    TEXT_MODELS = {
        'bert_indonesian': {
            'bert_name': 'cahya/bert-base-indonesian-522M',
            'fc_dim': 512,
            'max_len': 128,
            'batch_size': 32,
            'epochs': 10,
            'lr': 5e-5,
            's': 30,
            'margin': 0.5,
            'simple_mean': True,                # avg all tokens
            'loss_fn': 'arcface',
        },
        'bert_multilingual': {
            'bert_name': 'bert-base-multilingual-uncased',
            'fc_dim': 512,
            'max_len': 128,
            'batch_size': 32,
            'epochs': 10,
            'lr': 5e-5,
            's': 30,
            'margin': 0.5,
            'simple_mean': False,               # attention-weighted mean
            'loss_fn': 'arcface',
        },
        'xlm_roberta': {
            'bert_name': 'xlm-roberta-base',
            'fc_dim': 512,
            'max_len': 128,
            'batch_size': 32,
            'epochs': 10,
            'lr': 5e-5,
            's': 30,
            'margin': 0.5,
            'simple_mean': False,               # attention-weighted mean
            'loss_fn': 'arcface',
        },
    }

    # -- 2nd Stage: LightGBM Re-Ranker (matches inference engine) -------------
    # Trained as LambdaRank on FAISS top-K candidates with 9 cross-modal features
    LGBM_NUM_QUERIES = 5000          # products sampled for training data
    LGBM_K_CANDIDATES = 50          # FAISS candidates per query
    LGBM_PARAMS = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10],
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }
    LGBM_NUM_BOOST_ROUND = 200

    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
