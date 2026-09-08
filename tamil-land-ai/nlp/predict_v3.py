"""
TAMIL LAND RECORD NER - INDICBERT V3
=====================================

Robust inference script for the trained V3 model.

Features:
- Uses models/land-ner-v3
- Supports long Tamil paragraphs
- Uses overlapping inference windows
- Avoids tokenizer special-token backend incompatibility
- Handles subword tokens correctly
- Preserves character offsets
- Merges overlapping window predictions
- Extracts B-/I- entities
- Cleans punctuation from extracted values
- Produces structured JSON

Run:
    python nlp\predict_v3.py
"""

import os
import re
import json
import math
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "land-ner-v3",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

MAX_LENGTH = 256
OVERLAP = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "OWNER": "Owner",
    "FATHER_NAME": "Father Name",
    "SURVEY_NUMBER": "Survey Number",
    "PATTA_NUMBER": "Patta Number",
    "AREA": "Area",
    "BOUNDARY": "Boundary",
    "VILLAGE": "Village",
    "TALUK": "Taluk",
    "DISTRICT": "District",
    "YEAR": "Year",
    "DOCUMENT_NUMBER": "Document Number",
    "LAND_TYPE": "Land Type",
}


# ============================================================
# GLOBALS
# ============================================================

tokenizer = None
model = None

ID_TO_LABEL = {}


# ============================================================
# DEVICE INFORMATION
# ============================================================

def print_device_info():

    print("=" * 70)
    print("       TAMIL LAND RECORD NER - INDICBERT V3")
    print("=" * 70)
    print()

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(f"GPU: {gpu_name}")
        print(f"VRAM: {total_memory:.2f} GB")

        try:
            print(
                f"CUDA: {torch.version.cuda}"
            )
        except Exception:
            pass

    else:
        print("GPU: Not available")

    print()


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():

    global ID_TO_LABEL

    labels_file = os.path.join(
        MODEL_DIR,
        "labels.json"
    )

    if not os.path.exists(labels_file):

        raise FileNotFoundError(
            f"labels.json not found:\n{labels_file}"
        )

    with open(
        labels_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Expected V3 format
    # --------------------------------------------------------

    if "id_to_label" in data:

        ID_TO_LABEL = {
            int(k): v
            for k, v in data["id_to_label"].items()
        }

    elif "label_to_id" in data:

        ID_TO_LABEL = {
            int(v): k
            for k, v in data["label_to_id"].items()
        }

    elif "labels" in data:

        ID_TO_LABEL = {
            i: label
            for i, label in enumerate(data["labels"])
        }

    else:

        raise ValueError(
            "Could not find label vocabulary in labels.json"
        )

    print(
        f"Labels loaded: {len(ID_TO_LABEL)}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    global tokenizer

    print()
    print("Loading tokenizer...")

    # --------------------------------------------------------
    # Prefer the tokenizer saved with the trained model.
    # --------------------------------------------------------

    tokenizer_files_exist = (
        os.path.exists(
            os.path.join(
                MODEL_DIR,
                "tokenizer.json"
            )
        )
        and
        os.path.exists(
            os.path.join(
                MODEL_DIR,
                "tokenizer_config.json"
            )
        )
    )

    if tokenizer_files_exist:

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            use_fast=True,
        )

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )

    print("Tokenizer loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model

    print()
    print("Loading trained V3 NER model...")

    if not os.path.exists(MODEL_DIR):

        raise FileNotFoundError(
            f"Model directory does not exist:\n{MODEL_DIR}"
        )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR,
        num_labels=len(ID_TO_LABEL),
    )

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(text):

    """
    Tokenize WITHOUT special tokens.

    This is intentional.

    We do NOT ask the tokenizer to construct:
        [CLS] ... [SEP]

    because the installed tokenizer backend can fail with:

        TokenizersBackend has no attribute
        build_inputs_with_special_tokens

    Instead, we work entirely with raw tokens and add the
    model's special-token IDs manually for every window.
    """

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    return {
        "input_ids": input_ids,
        "offset_mapping": offsets,
    }


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(
    total_tokens,
    max_length=MAX_LENGTH,
    overlap=OVERLAP,
):

    # --------------------------------------------------------
    # Reserve two positions:
    #
    # [CLS] + content + [SEP]
    # --------------------------------------------------------

    content_size = max_length - 2

    if content_size <= 0:

        raise ValueError(
            "MAX_LENGTH must be greater than 2."
        )

    step = content_size - overlap

    if step <= 0:

        raise ValueError(
            "OVERLAP must be smaller than "
            "MAX_LENGTH - 2."
        )

    windows = []

    start = 0

    while start < total_tokens:

        end = min(
            start + content_size,
            total_tokens
        )

        windows.append(
            (
                start,
                end
            )
        )

        if end >= total_tokens:
            break

        start += step

    return windows


# ============================================================
# BUILD WINDOW
# ============================================================

def build_window(
    input_ids,
    offsets,
    start,
    end,
):

    content_ids = input_ids[start:end]

    content_offsets = offsets[start:end]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # --------------------------------------------------------
    # If tokenizer does not expose CLS/SEP, use special tokens
    # from the tokenizer vocabulary.
    # --------------------------------------------------------

    if cls_id is None:

        cls_token = tokenizer.cls_token

        if cls_token is not None:

            cls_id = tokenizer.convert_tokens_to_ids(
                cls_token
            )

    if sep_id is None:

        sep_token = tokenizer.sep_token

        if sep_token is not None:

            sep_id = tokenizer.convert_tokens_to_ids(
                sep_token
            )

    # --------------------------------------------------------
    # Some models may not have CLS/SEP.
    #
    # In that unlikely case, don't crash.
    # --------------------------------------------------------

    if cls_id is None:
        cls_id = tokenizer.pad_token_id

    if sep_id is None:
        sep_id = tokenizer.pad_token_id

    window_ids = (
        [cls_id]
        + content_ids
        + [sep_id]
    )

    window_offsets = (
        [(-1, -1)]
        + list(content_offsets)
        + [(-1, -1)]
    )

    attention_mask = [
        1
        for _ in window_ids
    ]

    return (
        window_ids,
        attention_mask,
        window_offsets,
    )


# ============================================================
# RUN ONE WINDOW
# ============================================================

@torch.no_grad()
def run_window(
    input_ids,
    attention_mask,
    offsets,
    text,
    global_start,
):

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    attention_tensor = torch.tensor(
        [attention_mask],
        dtype=torch.long,
        device=DEVICE,
    )

    outputs = model(
        input_ids=input_tensor,
        attention_mask=attention_tensor,
    )

    logits = outputs.logits[0]

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    predicted_ids = torch.argmax(
        probabilities,
        dim=-1,
    )

    predictions = []

    for local_index in range(
        len(input_ids)
    ):

        offset = offsets[local_index]

        start_char = offset[0]
        end_char = offset[1]

        # ----------------------------------------------------
        # Skip manually inserted CLS/SEP.
        # ----------------------------------------------------

        if start_char < 0 or end_char <= start_char:
            continue

        global_token_index = (
            global_start
            + local_index
            - 1
        )

        # ----------------------------------------------------
        # Convert model output to label.
        # ----------------------------------------------------

        label_id = int(
            predicted_ids[local_index].item()
        )

        label = ID_TO_LABEL.get(
            label_id,
            "O"
        )

        confidence = float(
            probabilities[
                local_index,
                label_id
            ].item()
        )

        token_text = text[
            start_char:end_char
        ]

        predictions.append(
            {
                "token_index": global_token_index,
                "start": start_char,
                "end": end_char,
                "token": token_text,
                "label": label,
                "confidence": confidence,
            }
        )

    return predictions


# ============================================================
# MERGE OVERLAPPING WINDOWS
# ============================================================

def merge_predictions(
    all_predictions
):

    """
    Merge duplicate token predictions created by overlapping
    windows.

    Each original tokenizer token has a character span.

    If a token occurs in multiple windows, retain the prediction
    with the highest confidence.
    """

    best = {}

    for prediction in all_predictions:

        key = (
            prediction["start"],
            prediction["end"],
        )

        old = best.get(key)

        if old is None:

            best[key] = prediction

        else:

            if (
                prediction["confidence"]
                > old["confidence"]
            ):

                best[key] = prediction

    result = list(
        best.values()
    )

    result.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    return result


# ============================================================
# NORMALIZE TOKEN TEXT
# ============================================================

def normalize_spaces(value):

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# CLEAN ENTITY VALUE
# ============================================================

def clean_entity_value(
    value,
    entity_type,
):

    value = normalize_spaces(value)

    # --------------------------------------------------------
    # Remove leading punctuation.
    # --------------------------------------------------------

    value = re.sub(
        r"^[\s,;:|]+",
        "",
        value
    )

    # --------------------------------------------------------
    # Remove trailing punctuation.
    # --------------------------------------------------------

    value = re.sub(
        r"[\s,;:|]+$",
        "",
        value
    )

    # --------------------------------------------------------
    # Don't allow sentence punctuation at the end.
    # --------------------------------------------------------

    value = re.sub(
        r"[.!?]+$",
        "",
        value
    )

    # --------------------------------------------------------
    # Survey numbers
    #
    # Example:
    #     72 / 4B
    #
    # becomes:
    #     72/4B
    # --------------------------------------------------------

    if entity_type == "SURVEY_NUMBER":

        value = re.sub(
            r"\s*/\s*",
            "/",
            value
        )

        value = re.sub(
            r"\s+",
            "",
            value
        )

    # --------------------------------------------------------
    # Patta/document numbers
    # --------------------------------------------------------

    elif entity_type in (
        "PATTA_NUMBER",
        "DOCUMENT_NUMBER",
    ):

        # Preserve slash and hyphen.
        value = re.sub(
            r"\s*([/-])\s*",
            r"\1",
            value
        )

    # --------------------------------------------------------
    # Area
    #
    # Preserve meaningful spaces:
    #
    # 1.75 ஏக்கர்
    # 3 ஏக்கர் 42 சென்ட்
    # 30 அடி
    # --------------------------------------------------------

    elif entity_type == "AREA":

        value = re.sub(
            r"\s*\.\s*",
            ".",
            value
        )

        value = normalize_spaces(
            value
        )

    return value.strip()


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(
    predictions,
    text,
):

    entities = []

    current = None

    for prediction in predictions:

        label = prediction["label"]

        if label == "O":

            if current is not None:

                entities.append(
                    current
                )

                current = None

            continue

        # ----------------------------------------------------
        # Parse B-/I-
        # ----------------------------------------------------

        if label.startswith("B-"):

            if current is not None:

                entities.append(
                    current
                )

            entity_type = label[2:]

            current = {
                "type": entity_type,
                "start": prediction["start"],
                "end": prediction["end"],
                "confidence_sum": prediction[
                    "confidence"
                ],
                "count": 1,
            }

        elif label.startswith("I-"):

            entity_type = label[2:]

            # ------------------------------------------------
            # Valid continuation.
            # ------------------------------------------------

            if (
                current is not None
                and current["type"]
                == entity_type
            ):

                current["end"] = max(
                    current["end"],
                    prediction["end"],
                )

                current[
                    "confidence_sum"
                ] += prediction[
                    "confidence"
                ]

                current["count"] += 1

            else:

                # --------------------------------------------
                # Broken I- sequence.
                #
                # Treat it as a new entity instead of
                # attaching it to an unrelated entity.
                # --------------------------------------------

                if current is not None:

                    entities.append(
                        current
                    )

                current = {
                    "type": entity_type,
                    "start": prediction["start"],
                    "end": prediction["end"],
                    "confidence_sum": prediction[
                        "confidence"
                    ],
                    "count": 1,
                }

    if current is not None:

        entities.append(
            current
        )

    # ========================================================
    # Convert spans into actual text
    # ========================================================

    final_entities = []

    for entity in entities:

        raw_value = text[
            entity["start"]:
            entity["end"]
        ]

        value = clean_entity_value(
            raw_value,
            entity["type"],
        )

        if not value:
            continue

        confidence = (
            entity["confidence_sum"]
            / max(
                entity["count"],
                1
            )
        )

        final_entities.append(
            {
                "type": entity["type"],
                "value": value,
                "confidence": confidence,
                "start": entity["start"],
                "end": entity["end"],
            }
        )

    return final_entities


# ============================================================
# TOKEN DISPLAY
# ============================================================

def print_token_predictions(
    predictions
):

    print()
    print("=" * 70)
    print("TOKEN PREDICTIONS")
    print("=" * 70)

    if not predictions:

        print("No token predictions.")

        return

    for prediction in predictions:

        token = prediction["token"]

        label = prediction["label"]

        confidence = (
            prediction["confidence"]
            * 100
        )

        print(
            f"{token:<28}"
            f"{label:<25}"
            f"{confidence:>7.2f}%"
        )


# ============================================================
# ENTITY DISPLAY
# ============================================================

def print_entities(
    entities
):

    print()
    print("=" * 70)
    print("EXTRACTED LAND RECORD")
    print("=" * 70)

    if not entities:

        print("No entities detected.")

        return

    for entity in entities:

        display_name = DISPLAY_NAMES.get(
            entity["type"],
            entity["type"]
        )

        confidence = (
            entity["confidence"]
            * 100
        )

        print(
            f"{display_name:<20}"
            f"→ {entity['value']} "
            f"({confidence:.1f}%)"
        )


# ============================================================
# STRUCTURED JSON
# ============================================================

def build_structured_json(
    entities
):

    structured = {
        "owner": [],
        "father_name": [],
        "patta_number": [],
        "survey_number": [],
        "area": [],
        "boundary": [],
        "village": [],
        "taluk": [],
        "district": [],
        "year": [],
        "document_number": [],
        "land_type": [],
    }

    for entity in entities:

        entity_type = entity["type"]

        key = entity_type.lower()

        if key not in structured:
            continue

        structured[key].append(
            {
                "value": entity["value"],
                "confidence": round(
                    entity["confidence"],
                    4
                ),
            }
        )

    return structured


# ============================================================
# PREDICTION
# ============================================================

def predict(text):

    text = text.strip()

    if not text:
        return None

    print()

    print(
        f"Input length: "
        f"{len(text)} characters"
    )

    # --------------------------------------------------------
    # Tokenize WITHOUT special tokens.
    # --------------------------------------------------------

    encoding = tokenize_text(
        text
    )

    input_ids = encoding[
        "input_ids"
    ]

    offsets = encoding[
        "offset_mapping"
    ]

    total_tokens = len(
        input_ids
    )

    print(
        f"Input tokens: {total_tokens}"
    )

    # --------------------------------------------------------
    # Create overlapping windows.
    # --------------------------------------------------------

    windows = create_windows(
        total_tokens,
        MAX_LENGTH,
        OVERLAP,
    )

    print(
        f"Inference windows: "
        f"{len(windows)}"
    )

    if not windows:

        print(
            "No usable tokens found."
        )

        return None

    # --------------------------------------------------------
    # Run each window.
    # --------------------------------------------------------

    all_predictions = []

    for window_number, (
        start,
        end
    ) in enumerate(
        windows,
        start=1
    ):

        print(
            f"Processing window "
            f"{window_number}/"
            f"{len(windows)}...",
            end="\r",
            flush=True,
        )

        (
            window_ids,
            attention_mask,
            window_offsets,
        ) = build_window(
            input_ids,
            offsets,
            start,
            end,
        )

        predictions = run_window(
            window_ids,
            attention_mask,
            window_offsets,
            text,
            start,
        )

        all_predictions.extend(
            predictions
        )

    print(
        " " * 70,
        end="\r"
    )

    # --------------------------------------------------------
    # Merge duplicates.
    # --------------------------------------------------------

    predictions = merge_predictions(
        all_predictions
    )

    # --------------------------------------------------------
    # Token predictions.
    # --------------------------------------------------------

    print_token_predictions(
        predictions
    )

    # --------------------------------------------------------
    # Extract entities.
    # --------------------------------------------------------

    entities = extract_entities(
        predictions,
        text,
    )

    print_entities(
        entities
    )

    # --------------------------------------------------------
    # JSON.
    # --------------------------------------------------------

    structured = build_structured_json(
        entities
    )

    print()
    print("=" * 70)
    print("STRUCTURED JSON")
    print("=" * 70)

    print(
        json.dumps(
            structured,
            ensure_ascii=False,
            indent=2,
        )
    )

    return structured


# ============================================================
# MAIN
# ============================================================

def main():

    print_device_info()

    print(
        f"Model directory: {MODEL_DIR}"
    )

    load_labels()

    print(
        f"Inference window: "
        f"{MAX_LENGTH} tokens"
    )

    print(
        f"Window overlap: "
        f"{OVERLAP} tokens"
    )

    load_tokenizer()

    load_model()

    print()
    print("=" * 70)
    print("       TAMIL LAND RECORD NER V3")
    print("=" * 70)
    print()

    print(
        "Enter Tamil land-record text."
    )

    print(
        "Long paragraphs are supported "
        "using overlapping windows."
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    while True:

        try:

            text = input(
                "Tamil text > "
            )

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            break

        if (
            text.strip().lower()
            == "exit"
        ):

            break

        if not text.strip():

            continue

        try:

            predict(text)

        except RuntimeError as e:

            if (
                "out of memory"
                in str(e).lower()
            ):

                print()
                print(
                    "CUDA OUT OF MEMORY."
                )

                print(
                    "Try reducing "
                    "MAX_LENGTH or OVERLAP."
                )

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

            else:

                print()
                print(
                    "Runtime error:"
                )

                print(e)

        except Exception as e:

            print()
            print(
                "Unexpected error:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()