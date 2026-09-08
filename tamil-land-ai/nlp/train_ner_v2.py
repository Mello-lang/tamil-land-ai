import os
import json
import random
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data", "ner_v2")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "land-ner-v2")

TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
VAL_FILE = os.path.join(DATA_DIR, "validation.json")
TEST_FILE = os.path.join(DATA_DIR, "test.json")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")

MAX_LENGTH = 256

# RTX 3050 4GB
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8

EPOCHS = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("       TAMIL LAND RECORD NER - DATASET V2")
print("=" * 70)
print()


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"VRAM: {memory:.2f} GB")

print()


# ============================================================
# CHECK FILES
# ============================================================

print("Checking dataset files...")

for path in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    LABELS_FILE,
]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    print(
        "  OK:",
        os.path.relpath(path, BASE_DIR)
    )

print()


# ============================================================
# LOAD LABELS
# ============================================================

print("Loading label vocabulary...")

with open(
    LABELS_FILE,
    "r",
    encoding="utf-8"
) as f:

    label_data = json.load(f)


# Your labels.json may contain:
# {"labels": [...]}
# OR
# {"label2id": {...}}
# OR
# {"0": "O", "1": "B-AREA", ...}

if isinstance(label_data, list):

    labels = label_data

elif "labels" in label_data:

    labels = label_data["labels"]

elif "label2id" in label_data:

    labels = [
        label
        for label, _ in sorted(
            label_data["label2id"].items(),
            key=lambda x: int(x[1])
        )
    ]

else:

    labels = [
        value
        for _, value in sorted(
            label_data.items(),
            key=lambda x: int(x[0])
        )
    ]


label2id = {
    label: i
    for i, label in enumerate(labels)
}

id2label = {
    i: label
    for i, label in enumerate(labels)
}


print()
print("Labels:")

for i, label in enumerate(labels):

    print(
        f"{i:2d} -> {label}"
    )

print()
print(
    f"Total labels: {len(labels)}"
)
print()


# ============================================================
# LOAD JSON DATA
# ============================================================

def load_json_dataset(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(
            f"{path} must contain a JSON list."
        )

    return Dataset.from_list(data)


print("Loading datasets...")

train_dataset = load_json_dataset(
    TRAIN_FILE
)

validation_dataset = load_json_dataset(
    VAL_FILE
)

test_dataset = load_json_dataset(
    TEST_FILE
)

print(
    f"Training examples:    {len(train_dataset)}"
)

print(
    f"Validation examples:  {len(validation_dataset)}"
)

print(
    f"Test examples:        {len(test_dataset)}"
)

print()


# ============================================================
# VALIDATE DATASET
# ============================================================

print("Validating dataset format...")

for split_name, dataset in [
    ("train", train_dataset),
    ("validation", validation_dataset),
    ("test", test_dataset),
]:

    if "tokens" not in dataset.column_names:

        raise RuntimeError(
            f"{split_name} dataset is missing 'tokens'."
        )

    if "labels" not in dataset.column_names:

        raise RuntimeError(
            f"{split_name} dataset is missing 'labels'."
        )

    for i in range(
        min(100, len(dataset))
    ):

        tokens = dataset[i]["tokens"]
        example_labels = dataset[i]["labels"]

        if len(tokens) != len(example_labels):

            raise ValueError(
                f"\nToken/label mismatch in "
                f"{split_name} example {i}\n"
                f"Tokens: {len(tokens)}\n"
                f"Labels: {len(example_labels)}"
            )

        for label in example_labels:

            if label not in label2id:

                raise ValueError(
                    f"Unknown label: {label}"
                )


print("Dataset format: OK")
print()


# ============================================================
# SHOW EXAMPLE
# ============================================================

print("Example dataset record:")
print("-" * 70)

example = train_dataset[0]

print(
    "TOKENS:",
    example["tokens"]
)

print(
    "LABELS:",
    example["labels"]
)

print("-" * 70)
print()


# ============================================================
# TOKENIZER
# ============================================================

print("Loading IndicBERT v2 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print(
    "Tokenizer loaded successfully."
)

print()


# ============================================================
# TOKENIZE + ALIGN LABELS
# ============================================================

def tokenize_and_align_labels(examples):

    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []

    for tokens, word_labels in zip(
        examples["tokens"],
        examples["labels"]
    ):

        # ----------------------------------------------------
        # Tokenize each original dataset token separately.
        #
        # This is important because the dataset labels are
        # already aligned with the "tokens" array.
        # ----------------------------------------------------

        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        word_ids = encoding.word_ids()

        aligned_labels = []

        previous_word_id = None

        for word_id in word_ids:

            # Special token
            if word_id is None:

                aligned_labels.append(-100)

                continue

            original_label = word_labels[word_id]

            # ------------------------------------------------
            # First sub-token
            # ------------------------------------------------

            if word_id != previous_word_id:

                aligned_labels.append(
                    label2id[original_label]
                )

            # ------------------------------------------------
            # Additional sub-token
            # ------------------------------------------------

            else:

                # B-ENTITY -> I-ENTITY
                if original_label.startswith("B-"):

                    entity_type = original_label[2:]

                    continuation = (
                        "I-" + entity_type
                    )

                    if continuation in label2id:

                        aligned_labels.append(
                            label2id[continuation]
                        )

                    else:

                        aligned_labels.append(
                            label2id[original_label]
                        )

                else:

                    aligned_labels.append(
                        label2id[original_label]
                    )

            previous_word_id = word_id

        batch_input_ids.append(
            encoding["input_ids"]
        )

        batch_attention_masks.append(
            encoding["attention_mask"]
        )

        batch_labels.append(
            aligned_labels
        )

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_masks,
        "labels": batch_labels,
    }


# ============================================================
# TOKENIZE DATA
# ============================================================

print("Tokenizing training data...")

tokenized_train = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
)

print("Tokenizing validation data...")

tokenized_validation = validation_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=validation_dataset.column_names,
    desc="Tokenizing validation",
)

print("Tokenizing test data...")

tokenized_test = test_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=test_dataset.column_names,
    desc="Tokenizing test",
)

print()
print("Tokenization complete.")
print()


# ============================================================
# CHECK TOKENIZED DATA
# ============================================================

print("Checking tokenized dataset...")

print(
    "Columns:",
    tokenized_train.column_names
)

required = [
    "input_ids",
    "attention_mask",
    "labels",
]

for column in required:

    if column not in tokenized_train.column_names:

        raise RuntimeError(
            f"Missing tokenized column: {column}"
        )

print(
    "Tokenized dataset: OK"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading IndicBERT v2...")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,

    num_labels=len(labels),

    id2label=id2label,

    label2id=label2id,

    trust_remote_code=True,

    ignore_mismatched_sizes=True,
)

print(
    "Model loaded successfully."
)

print()


# ============================================================
# PARAMETER COUNT
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print()


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels_array = eval_pred

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []
    true_labels = []

    for prediction_row, label_row in zip(
        predictions,
        labels_array
    ):

        pred = []
        gold = []

        for prediction, label in zip(
            prediction_row,
            label_row
        ):

            if label == -100:
                continue

            pred.append(
                int(prediction)
            )

            gold.append(
                int(label)
            )

        true_predictions.append(pred)
        true_labels.append(gold)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    correct = 0
    total = 0

    for pred, gold in zip(
        true_predictions,
        true_labels
    ):

        for p, g in zip(pred, gold):

            total += 1

            if p == g:
                correct += 1

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    # --------------------------------------------------------
    # BIO entity extraction
    # --------------------------------------------------------

    def get_entities(sequence):

        entities = []

        current_type = None
        start = None

        for index, label_id in enumerate(sequence):

            label = id2label[int(label_id)]

            if label == "O":

                if current_type is not None:

                    entities.append(
                        (
                            start,
                            index - 1,
                            current_type
                        )
                    )

                    current_type = None
                    start = None

                continue

            if label.startswith("B-"):

                if current_type is not None:

                    entities.append(
                        (
                            start,
                            index - 1,
                            current_type
                        )
                    )

                current_type = label[2:]
                start = index

            elif label.startswith("I-"):

                entity_type = label[2:]

                if (
                    current_type is None
                    or current_type != entity_type
                ):

                    if current_type is not None:

                        entities.append(
                            (
                                start,
                                index - 1,
                                current_type
                            )
                        )

                    current_type = entity_type
                    start = index

        if current_type is not None:

            entities.append(
                (
                    start,
                    len(sequence) - 1,
                    current_type
                )
            )

        return set(entities)

    true_positive = 0
    predicted_total = 0
    actual_total = 0

    for pred, gold in zip(
        true_predictions,
        true_labels
    ):

        pred_entities = get_entities(pred)
        gold_entities = get_entities(gold)

        true_positive += len(
            pred_entities & gold_entities
        )

        predicted_total += len(
            pred_entities
        )

        actual_total += len(
            gold_entities
        )

    precision = (
        true_positive / predicted_total
        if predicted_total
        else 0.0
    )

    recall = (
        true_positive / actual_total
        if actual_total
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print("Configuring training...")
print()

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    logging_strategy="steps",

    logging_steps=50,

    fp16=torch.cuda.is_available(),

    dataloader_pin_memory=torch.cuda.is_available(),

    dataloader_num_workers=0,

    save_total_limit=2,

    report_to="none",

    seed=SEED,

    remove_unused_columns=True,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# START TRAINING
# ============================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Training examples: {len(train_dataset)}"
)

print(
    f"Validation examples: {len(validation_dataset)}"
)

print(
    f"Test examples: {len(test_dataset)}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Gradient accumulation: "
    f"{GRADIENT_ACCUMULATION}"
)

print(
    f"Effective batch size: "
    f"{BATCH_SIZE * GRADIENT_ACCUMULATION}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print()


# ============================================================
# TRAIN
# ============================================================

train_result = trainer.train()


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print()

print("Saving model...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    f"Model saved to:\n{OUTPUT_DIR}"
)

print()


# ============================================================
# VALIDATION
# ============================================================

print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

validation_metrics = trainer.evaluate(
    tokenized_validation
)

for key, value in validation_metrics.items():

    print(
        f"{key}: {value}"
    )

print()


# ============================================================
# TEST
# ============================================================

print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

test_metrics = trainer.evaluate(
    tokenized_test,
    metric_key_prefix="test"
)

for key, value in test_metrics.items():

    print(
        f"{key}: {value}"
    )

print()


# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

evaluation_metrics = {
    "validation": validation_metrics,
    "test": test_metrics,
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "evaluation_metrics.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_metrics,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE LABELS
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "labels.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "label2id": label2id,
            "id2label": {
                str(k): v
                for k, v in id2label.items()
            },
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE TRAINING METRICS
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "training_metrics.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        train_result.metrics,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print("=" * 70)
print("MODEL TRAINING FINISHED SUCCESSFULLY")
print("=" * 70)
print()

print(
    "Output directory:"
)

print(
    f"  {OUTPUT_DIR}"
)

print()

print(
    "Next command:"
)

print(
    "  python nlp\\predict.py"
)

print()