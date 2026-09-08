import json
import os
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# TAMIL LAND RECORD NER TRAINER
# IndicBERT v2 + BIO NER
# RTX 3050 4 GB optimized
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

TRAIN_FILE = "data/ner/train.json"
VALIDATION_FILE = "data/ner/validation.json"
TEST_FILE = "data/ner/test.json"

OUTPUT_DIR = "models/land-ner"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION = 4

EPOCHS = 3

LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.01


# ============================================================
# START
# ============================================================

print("=" * 70)
print("       TAMIL LAND RECORD NER - INDICBERT v2")
print("=" * 70)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("\nDevice: CUDA")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    gpu_memory = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"VRAM: {gpu_memory:.2f} GB"
    )

else:

    DEVICE = torch.device("cpu")

    print("\nWARNING: CUDA is not available.")
    print("Training will run on CPU.")


# ============================================================
# CHECK DATASET FILES
# ============================================================

print("\nChecking dataset files...")

required_files = [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nDataset file not found:\n{file_path}\n"
            f"Current directory:\n{os.getcwd()}"
        )

    print(
        f"  OK: {file_path}"
    )


# ============================================================
# LOAD JSON
# ============================================================

def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


print("\nLoading datasets...")

train_data = load_json(
    TRAIN_FILE
)

validation_data = load_json(
    VALIDATION_FILE
)

test_data = load_json(
    TEST_FILE
)

print(
    "Training examples:   ",
    len(train_data)
)

print(
    "Validation examples: ",
    len(validation_data)
)

print(
    "Test examples:       ",
    len(test_data)
)


# ============================================================
# VALIDATE DATASET FORMAT
# ============================================================

print("\nValidating dataset format...")


def validate_dataset(data, name):

    if not isinstance(data, list):

        raise ValueError(
            f"{name} must contain a JSON list."
        )

    for index, example in enumerate(data):

        if "tokens" not in example:

            raise ValueError(
                f"{name} example {index} "
                f"is missing 'tokens'."
            )

        if "ner_tags" not in example:

            raise ValueError(
                f"{name} example {index} "
                f"is missing 'ner_tags'."
            )

        if len(example["tokens"]) != len(
            example["ner_tags"]
        ):

            raise ValueError(
                f"{name} example {index} has "
                f"{len(example['tokens'])} tokens but "
                f"{len(example['ner_tags'])} labels."
            )


validate_dataset(
    train_data,
    "Training dataset"
)

validate_dataset(
    validation_data,
    "Validation dataset"
)

validate_dataset(
    test_data,
    "Test dataset"
)

print(
    "Dataset format: OK"
)


# ============================================================
# CREATE HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_list(
    train_data
)

validation_dataset = Dataset.from_list(
    validation_data
)

test_dataset = Dataset.from_list(
    test_data
)


# ============================================================
# BUILD ENTITY VOCABULARY
# ============================================================

print("\nBuilding label vocabulary...")


entity_types = set()


for dataset in [
    train_data,
    validation_data,
    test_data,
]:

    for example in dataset:

        for tag in example["ner_tags"]:

            if tag == "O":

                continue

            if "-" not in tag:

                raise ValueError(
                    f"Invalid BIO tag: {tag}"
                )

            prefix, entity_type = tag.split(
                "-",
                1
            )

            if prefix not in [
                "B",
                "I",
            ]:

                raise ValueError(
                    f"Invalid BIO prefix: {tag}"
                )

            entity_types.add(
                entity_type
            )


entity_types = sorted(
    entity_types
)


# ============================================================
# CREATE COMPLETE BIO LABEL LIST
# ============================================================

label_list = ["O"]


for entity_type in entity_types:

    label_list.append(
        f"B-{entity_type}"
    )

    label_list.append(
        f"I-{entity_type}"
    )


label2id = {
    label: index
    for index, label in enumerate(
        label_list
    )
}


id2label = {
    index: label
    for index, label in enumerate(
        label_list
    )
}


# ============================================================
# PRINT LABELS
# ============================================================

print("\nLabels:")

for index, label in enumerate(
    label_list
):

    print(
        f"{index:2d} -> {label}"
    )


print(
    "\nTotal labels:",
    len(label_list)
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print(
    "\nLoading IndicBERT v2 tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(
    "Tokenizer loaded successfully."
)


# ============================================================
# TOKENIZE + ALIGN BIO LABELS
# ============================================================

def tokenize_and_align_labels(
    examples
):

    tokenized_inputs = tokenizer(

        examples["tokens"],

        is_split_into_words=True,

        truncation=True,

        max_length=MAX_LENGTH,

        padding=False,

    )


    all_labels = []


    for batch_index, labels in enumerate(
        examples["ner_tags"]
    ):

        word_ids = (
            tokenized_inputs.word_ids(
                batch_index
            )
        )


        previous_word_id = None

        label_ids = []


        for word_id in word_ids:

            # ------------------------------------------------
            # SPECIAL TOKENS
            # ------------------------------------------------

            if word_id is None:

                label_ids.append(
                    -100
                )


            # ------------------------------------------------
            # FIRST SUBWORD
            # ------------------------------------------------

            elif word_id != previous_word_id:

                original_label = labels[
                    word_id
                ]

                label_ids.append(
                    label2id[
                        original_label
                    ]
                )


            # ------------------------------------------------
            # CONTINUATION SUBWORD
            # ------------------------------------------------

            else:

                original_label = labels[
                    word_id
                ]


                # B-ENTITY becomes I-ENTITY
                # for subsequent subwords.

                if original_label.startswith(
                    "B-"
                ):

                    entity_type = (
                        original_label[2:]
                    )

                    continuation_label = (
                        f"I-{entity_type}"
                    )

                else:

                    continuation_label = (
                        original_label
                    )


                label_ids.append(
                    label2id[
                        continuation_label
                    ]
                )


            previous_word_id = word_id


        all_labels.append(
            label_ids
        )


    tokenized_inputs["labels"] = (
        all_labels
    )


    return tokenized_inputs


# ============================================================
# TOKENIZE TRAINING DATA
# ============================================================

print(
    "\nTokenizing training data..."
)

tokenized_train = train_dataset.map(

    tokenize_and_align_labels,

    batched=True,

    remove_columns=(
        train_dataset.column_names
    ),

    desc="Tokenizing train",
)


# ============================================================
# TOKENIZE VALIDATION DATA
# ============================================================

print(
    "Tokenizing validation data..."
)

tokenized_validation = (
    validation_dataset.map(

        tokenize_and_align_labels,

        batched=True,

        remove_columns=(
            validation_dataset.column_names
        ),

        desc="Tokenizing validation",
    )
)


# ============================================================
# TOKENIZE TEST DATA
# ============================================================

print(
    "Tokenizing test data..."
)

tokenized_test = test_dataset.map(

    tokenize_and_align_labels,

    batched=True,

    remove_columns=(
        test_dataset.column_names
    ),

    desc="Tokenizing test",
)


print(
    "\nTokenization complete."
)


# ============================================================
# VERIFY TOKENIZED DATA
# ============================================================

print(
    "\nChecking tokenized dataset..."
)

print(
    "Columns:",
    tokenized_train.column_names
)


# IndicBERT v2 does not use token_type_ids.
# Only these columns are required for our NER model.

expected_columns = [
    "input_ids",
    "attention_mask",
    "labels",
]


for column in expected_columns:

    if column not in tokenized_train.column_names:

        raise RuntimeError(
            f"Expected column missing: {column}"
        )


# Make sure raw dataset columns were removed.

if "tokens" in tokenized_train.column_names:

    raise RuntimeError(
        "Raw 'tokens' column was not removed."
    )


if "ner_tags" in tokenized_train.column_names:

    raise RuntimeError(
        "Raw 'ner_tags' column was not removed."
    )


print(
    "Tokenized dataset: OK"
)

# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading IndicBERT v2..."
)

model = AutoModelForTokenClassification.from_pretrained(

    MODEL_NAME,

    num_labels=len(label_list),

    id2label=id2label,

    label2id=label2id,

)


# ============================================================
# MOVE MODEL TO GPU
# ============================================================

model.to(
    DEVICE
)


print(
    "Model loaded successfully."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    "\nTotal parameters:",
    f"{total_parameters:,}"
)


print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = (
    DataCollatorForTokenClassification(

        tokenizer=tokenizer,

        padding=True,

        return_tensors="pt",

    )
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_prediction
):

    predictions, labels = (
        eval_prediction
    )


    # Convert logits to label IDs

    predictions = predictions.argmax(
        axis=-1
    )


    true_predictions = []

    true_labels = []


    for prediction, label in zip(
        predictions,
        labels
    ):

        current_predictions = []

        current_labels = []


        for pred, lab in zip(
            prediction,
            label
        ):

            # Ignore special tokens

            if lab == -100:

                continue


            current_predictions.append(
                id2label[int(pred)]
            )


            current_labels.append(
                id2label[int(lab)]
            )


        true_predictions.append(
            current_predictions
        )


        true_labels.append(
            current_labels
        )


    precision = precision_score(

        true_labels,

        true_predictions,

        zero_division=0,

    )


    recall = recall_score(

        true_labels,

        true_predictions,

        zero_division=0,

    )


    f1 = f1_score(

        true_labels,

        true_predictions,

        zero_division=0,

    )


    accuracy = accuracy_score(

        true_labels,

        true_predictions,

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

print(
    "\nConfiguring training..."
)


training_args = TrainingArguments(

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir=OUTPUT_DIR,


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    eval_strategy="epoch",

    save_strategy="epoch",


    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),


    # --------------------------------------------------------
    # GRADIENT ACCUMULATION
    # --------------------------------------------------------

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION
    ),


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,


    # --------------------------------------------------------
    # RTX 3050 FP16
    # --------------------------------------------------------

    fp16=torch.cuda.is_available(),


    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=50,


    # --------------------------------------------------------
    # CHECKPOINTS
    # --------------------------------------------------------

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,


    # --------------------------------------------------------
    # DISABLE WANDB ETC.
    # --------------------------------------------------------

    report_to="none",


    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------

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

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print("STARTING TRAINING")

print("=" * 70)

print()


print(
    "Model:",
    MODEL_NAME
)


print(
    "Training examples:",
    len(tokenized_train)
)


print(
    "Validation examples:",
    len(tokenized_validation)
)


print(
    "Test examples:",
    len(tokenized_test)
)


print(
    "Labels:",
    len(label_list)
)


print(
    "Epochs:",
    EPOCHS
)


print(
    "Batch size:",
    TRAIN_BATCH_SIZE
)


print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION
)


print(
    "Effective batch size:",
    TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION
)


print(
    "Learning rate:",
    LEARNING_RATE
)


print(
    "Max sequence length:",
    MAX_LENGTH
)


print(
    "FP16:",
    torch.cuda.is_available()
)


print()


# ============================================================
# CLEAR GPU CACHE
# ============================================================

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# TRAIN
# ============================================================

train_result = trainer.train()


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving trained model..."
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


trainer.save_model(
    OUTPUT_DIR
)


tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# SAVE LABEL MAP
# ============================================================

label_map = {

    "labels": label_list,

    "label2id": label2id,

    "id2label": {
        str(key): value
        for key, value in id2label.items()
    },

}


with open(

    os.path.join(
        OUTPUT_DIR,
        "labels.json"
    ),

    "w",

    encoding="utf-8",

) as file:

    json.dump(

        label_map,

        file,

        ensure_ascii=False,

        indent=2,

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

    encoding="utf-8",

) as file:

    json.dump(

        train_result.metrics,

        file,

        indent=2,

    )


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print(
    "\nEvaluating validation dataset..."
)


validation_metrics = trainer.evaluate()


print(
    "\nValidation results:"
)


for key, value in validation_metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# TEST EVALUATION
# ============================================================

print(
    "\nEvaluating test dataset..."
)


test_metrics = trainer.evaluate(

    eval_dataset=tokenized_test,

    metric_key_prefix="test",

)


print(
    "\nTest results:"
)


for key, value in test_metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

with open(

    os.path.join(
        OUTPUT_DIR,
        "evaluation_metrics.json"
    ),

    "w",

    encoding="utf-8",

) as file:

    json.dump(

        {
            "validation": validation_metrics,
            "test": test_metrics,
        },

        file,

        indent=2,

    )


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated(0)
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved(0)
        / (1024 ** 3)
    )

    print(
        "\nGPU memory allocated:",
        f"{allocated:.2f} GB"
    )

    print(
        "GPU memory reserved:",
        f"{reserved:.2f} GB"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")

print("=" * 70)

print("TRAINING COMPLETE")

print("=" * 70)

print()


print(
    "Model saved to:"
)


print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


print()


print(
    "Files created:"
)


print(
    "  - model files"
)


print(
    "  - tokenizer files"
)


print(
    "  - labels.json"
)


print(
    "  - training_metrics.json"
)


print(
    "  - evaluation_metrics.json"
)


print()


print(
    "Next step: nlp/predict.py"
)


print("=" * 70)