"""Create local synthetic assets and executable YAMLs without downloading a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast


def create_smoke_inputs(*, output_dir: str | Path) -> Path:
    """Write one tiny base encoder, disjoint corpora, predicted spans and all policy YAMLs."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "severe",
        "chronic",
        "heart",
        "failure",
        "without",
        "pulmonary",
        "edema",
        "no",
        "today",
        "patient",
        "stable",
    ]
    backend = Tokenizer(
        models.WordPiece(
            vocab={word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    torch.manual_seed(42)
    model = BertForMaskedLM(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(root / "base")
    tokenizer.save_pretrained(root / "base")
    text = "severe chronic heart failure without pulmonary edema"
    training = pd.DataFrame(
        [
            {
                "doc_id": f"train-{index}",
                "text": text + " no",
                "source": "synthetic",
                "translation_status": "native",
            }
            for index in range(4)
        ]
    )
    validation = pd.DataFrame([{"doc_id": "dev", "text": text + " patient stable"}])
    predictions = pd.DataFrame(
        [
            {
                "filename": row.doc_id,
                "label": "DISEASE",
                "start_span": 15,
                "end_span": 28,
                "text": "heart failure",
                "score": 0.8,
                "code": "C1",
            }
            for row in training.itertuples()
        ]
    )
    for name, frame in (("train", training), ("dev", validation), ("predictions", predictions)):
        frame.to_parquet(root / f"{name}.parquet", index=False)
    parameters = {"epochs": 1, "batch_size": 2, "evaluation_probability": 0.5}
    common = {
        "documents": str(root / "train.parquet"),
        "validation": str(root / "dev.parquet"),
        "max_length": 32,
        "training_parameters": parameters,
    }
    configs = {
        "original": {
            "task": "cpt.evaluate_checkpoint",
            "base_model": str(root / "base"),
            "documents": str(root / "dev.parquet"),
            "probability": 0.5,
            "output_dir": str(root / "original-evaluation"),
        },
        "clm": {
            "task": "cpt.pretrain_causal",
            "base_model": str(root / "base"),
            "output_dir": str(root / "clm"),
            **common,
        },
        "mlm-only": {
            "task": "cpt.pretrain_masked",
            "base_model": str(root / "base"),
            "output_dir": str(root / "mlm-only"),
            "mask_policy": {"policy": "M0", "probability": 0.5},
            **common,
        },
    }
    for index in range(9):
        configs[f"M{index}"] = {
            "task": "cpt.pretrain_masked",
            "base_model": str(root / "clm/checkpoints/epoch_001/model"),
            "output_dir": str(root / f"M{index}"),
            "entity_predictions": str(root / "predictions.parquet"),
            "mask_policy": {
                "policy": f"M{index}",
                "probability": 0.5,
                "adaptive_enabled": index == 8,
            },
            **common,
        }
    for name, config in configs.items():
        (root / f"{name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    print(create_smoke_inputs(output_dir=arguments.output_dir))
