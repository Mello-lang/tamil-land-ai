"""
Train IndicBERTv2 Tamil Land Record NER V4.

Training-only:
- Uses the V4 dataset generator output.
- Handles wordpiece label alignment correctly.
- Evaluates exact entity spans, not confidence.
- Saves the best validation-F1 checkpoint.
- Uses gradient accumulation for a 4 GB GPU.
"""

from __future__ import annotations

import json
import random
import inspect
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "ner_v4"
OUT = BASE / "models" / "land-ner-v4"

MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8

EPOCHS = 5
LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
SEED = 20260909

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_dataset(path: Path) -> Dataset:
    return Dataset.from_list(
        json.loads(path.read_text(encoding="utf-8"))
    )


def entity_spans(label_ids, id2label):
    """
    Convert BIO labels into exact token-span tuples:
        (start_token, end_token_exclusive, entity_type)
    """
    entities = []
    current_type = None
    start = None

    for i, label_id in enumerate(label_ids):
        label = id2label[int(label_id)]

        if label == "O":
            if current_type is not None:
                entities.append((start, i, current_type))
                current_type = None
                start = None
            continue

        if label.startswith("B-"):
            if current_type is not None:
                entities.append((start, i, current_type))
            current_type = label[2:]
            start = i
            continue

        if label.startswith("I-"):
            incoming_type = label[2:]
            if current_type != incoming_type:
                if current_type is not None:
                    entities.append((start, i, current_type))
                current_type = incoming_type
                start = i

    if current_type is not None:
        entities.append((start, len(label_ids), current_type))

    return set(entities)


def build_metrics(id2label):
    def compute_metrics(eval_prediction):
        predictions, gold_labels = eval_prediction
        predictions = np.argmax(predictions, axis=-1)

        true_positive = 0
        predicted_total = 0
        gold_total = 0

        per_type = {
            entity: {"tp": 0, "pred": 0, "gold": 0}
            for entity in id2label.values()
            if entity != "O" and not entity.startswith("B-") and not entity.startswith("I-")
        }

        # Rebuild entity names from BIO labels.
        entity_types = sorted({
            label.split("-", 1)[1]
            for label in id2label.values()
            if label.startswith("B-")
        })
        per_type = {e: {"tp": 0, "pred": 0, "gold": 0} for e in entity_types}

        for pred_row, gold_row in zip(predictions, gold_labels):
            pred_ids = [
                int(p)
                for p, g in zip(pred_row, gold_row)
                if int(g) != -100
            ]
            gold_ids = [
                int(g)
                for g in gold_row
                if int(g) != -100
            ]

            pred_entities = entity_spans(pred_ids, id2label)
            gold_entities = entity_spans(gold_ids, id2label)

            true_positive += len(pred_entities & gold_entities)
            predicted_total += len(pred_entities)
            gold_total += len(gold_entities)

            for entity in entity_types:
                p = {x for x in pred_entities if x[2] == entity}
                g = {x for x in gold_entities if x[2] == entity}
                per_type[entity]["tp"] += len(p & g)
                per_type[entity]["pred"] += len(p)
                per_type[entity]["gold"] += len(g)

        precision = (
            true_positive / predicted_total
            if predicted_total else 0.0
        )
        recall = (
            true_positive / gold_total
            if gold_total else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        result = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # Macro F1 prevents large/easy labels from hiding weak numeric labels.
        label_f1 = []
        for entity, stats in per_type.items():
            p = stats["tp"] / stats["pred"] if stats["pred"] else 0.0
            r = stats["tp"] / stats["gold"] if stats["gold"] else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            result[f"f1_{entity.lower()}"] = f
            label_f1.append(f)

        result["macro_f1"] = (
            sum(label_f1) / len(label_f1)
            if label_f1 else 0.0
        )

        return result

    return compute_metrics


def main():
    required = [
        DATA / "train.json",
        DATA / "validation.json",
        DATA / "test.json",
        DATA / "labels.json",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run generate_dataset_v4.py first."
            )

    label_data = json.loads(
        (DATA / "labels.json").read_text(encoding="utf-8")
    )
    labels = label_data["labels"]

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    train_data = load_dataset(DATA / "train.json")
    val_data = load_dataset(DATA / "validation.json")
    test_data = load_dataset(DATA / "test.json")

    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Labels: {len(labels)}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    def align_batch(examples):
        encoded = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        aligned_labels = []

        for batch_index in range(len(examples["tokens"])):
            word_ids = encoded.word_ids(batch_index=batch_index)
            source_labels = examples["labels"][batch_index]

            result = []
            previous_word = None

            for word_id in word_ids:
                if word_id is None:
                    result.append(-100)
                    continue

                source_label = source_labels[word_id]

                if word_id != previous_word:
                    result.append(label2id[source_label])
                else:
                    # Subsequent wordpieces inherit I-ENTITY when possible.
                    if source_label.startswith("B-"):
                        entity = source_label[2:]
                        result.append(label2id.get(
                            f"I-{entity}",
                            label2id[source_label],
                        ))
                    else:
                        result.append(label2id[source_label])

                previous_word = word_id

            aligned_labels.append(result)

        encoded["labels"] = aligned_labels
        return encoded

    train_encoded = train_data.map(
        align_batch,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizing train",
    )

    val_encoded = val_data.map(
        align_batch,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Tokenizing validation",
    )

    test_encoded = test_data.map(
        align_batch,
        batched=True,
        remove_columns=test_data.column_names,
        desc="Tokenizing test",
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    compute_metrics = build_metrics(id2label)

    # Your installed Transformers build is older/different from the API
    # used by the current examples. Build arguments from the parameters that
    # actually exist in the installed TrainingArguments class.
    params = inspect.signature(TrainingArguments.__init__).parameters

    common_args = {
        "output_dir": str(OUT),
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "fp16": torch.cuda.is_available(),
        "seed": SEED,
        "dataloader_num_workers": 0,
    }

    # Warmup ratio is deliberately avoided because this installed
    # TrainingArguments does not expose it. Use warmup_steps instead.
    # 10% is approximately 2,000 steps for this V4 training run.
    if "warmup_steps" in params:
        common_args["warmup_steps"] = 2000

    if "report_to" in params:
        common_args["report_to"] = "none"

    # Transformers renamed evaluation_strategy -> eval_strategy.
    if "eval_strategy" in params:
        common_args["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        common_args["evaluation_strategy"] = "epoch"

    if "save_strategy" in params:
        common_args["save_strategy"] = "epoch"
    elif "save_steps" in params:
        # Fallback for old releases that only support step-based saving.
        common_args["save_steps"] = 500

    # Only pass parameters accepted by the installed version.
    supported_args = {
        key: value
        for key, value in common_args.items()
        if key in params
    }

    # If evaluation/saving strategy arguments are unavailable, disable
    # load_best_model_at_end because it depends on periodic evaluation.
    has_eval_strategy = (
        "eval_strategy" in supported_args
        or "evaluation_strategy" in supported_args
    )
    has_save_strategy = "save_strategy" in supported_args or "save_steps" in supported_args

    if not (has_eval_strategy and has_save_strategy):
        supported_args.pop("load_best_model_at_end", None)
        supported_args.pop("metric_for_best_model", None)
        supported_args.pop("greater_is_better", None)

    args = TrainingArguments(**supported_args)

    print("TrainingArguments compatibility mode:")
    print("  Accepted parameters:", ", ".join(sorted(supported_args.keys())))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting V4 training...\n")
    trainer.train()

    print("\n===== VALIDATION =====")
    validation_result = trainer.evaluate(val_encoded)
    print(validation_result)

    print("\n===== TEST =====")
    test_result = trainer.evaluate(test_encoded)
    print(test_result)

    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))

    (OUT / "labels.json").write_text(
        json.dumps(
            label_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nV4 model saved to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
