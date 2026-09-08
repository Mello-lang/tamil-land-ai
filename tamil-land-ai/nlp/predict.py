import json
import os
import re

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# ============================================================
# TAMIL LAND RECORD NER - PREDICTION
# IndicBERT v2
# ============================================================

MODEL_DIR = "models/land-ner"
MAX_LENGTH = 128


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("Device: CUDA")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = torch.device("cpu")

    print("Device: CPU")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isdir(MODEL_DIR):

    raise FileNotFoundError(
        f"Model directory not found: {MODEL_DIR}"
    )


model_file = os.path.join(
    MODEL_DIR,
    "model.safetensors"
)

if not os.path.exists(model_file):

    raise FileNotFoundError(
        f"Model file not found: {model_file}"
    )


# ============================================================
# LOAD LABELS
# ============================================================

labels_file = os.path.join(
    MODEL_DIR,
    "labels.json"
)

with open(
    labels_file,
    "r",
    encoding="utf-8"
) as file:

    label_data = json.load(file)


label_list = label_data["labels"]

label2id = label_data["label2id"]

id2label = {
    int(key): value
    for key, value
    in label_data["id2label"].items()
}


print(
    "\nLabels loaded:",
    len(label_list)
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

print(
    "Tokenizer loaded."
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading trained NER model..."
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# ENTITY NAMES
# ============================================================

ENTITY_NAMES = {

    "AREA":
        "Area",

    "BOUNDARY":
        "Boundary",

    "DISTRICT":
        "District",

    "DOCUMENT_NUMBER":
        "Document Number",

    "FATHER_NAME":
        "Father Name",

    "LAND_TYPE":
        "Land Type",

    "OWNER":
        "Owner",

    "PATTA_NUMBER":
        "Patta Number",

    "SURVEY_NUMBER":
        "Survey Number",

    "TALUK":
        "Taluk",

    "VILLAGE":
        "Village",

    "YEAR":
        "Year",

}


# ============================================================
# LABEL HELPERS
# ============================================================

def split_label(label):

    if label == "O":

        return "O", None

    if "-" not in label:

        return label, None

    prefix, entity_type = label.split(
        "-",
        1
    )

    return prefix, entity_type


# ============================================================
# TOKEN TEXT RECONSTRUCTION
# ============================================================

def clean_token_text(token):

    """
    Clean tokenizer artifacts.

    SentencePiece/BPE tokenizers can return
    pieces such as whitespace markers.

    We don't blindly add spaces because Tamil,
    punctuation and numbers need special handling.
    """

    if token is None:

        return ""

    token = str(token)

    # Common SentencePiece marker

    token = token.replace(
        "▁",
        " "
    )

    # Common BERT-style word marker

    token = token.replace(
        "##",
        ""
    )

    return token


def should_add_space(
    previous_text,
    current_text,
    previous_end,
    current_start
):

    """
    Decide whether two adjacent tokenizer pieces
    need a space when reconstructing an entity.
    """

    if not previous_text:

        return False

    if not current_text:

        return False

    # If there was actual whitespace in the original text,
    # preserve it.

    if current_start > previous_end:

        original_gap = current_start - previous_end

        if original_gap > 0:

            return True

    # Never put spaces before punctuation.

    if current_text in [
        ".",
        ",",
        ":",
        ";",
        "/",
        "-",
        ")",
        "]",
        "}",
        "%",
    ]:

        return False

    # Never put spaces after opening punctuation.

    if previous_text in [
        "(",
        "[",
        "{",
        "/",
        "-",
    ]:

        return False

    # Numbers split into multiple tokenizer pieces
    # must stay together.

    if (
        previous_text[-1:].isdigit()
        and current_text[:1].isdigit()
    ):

        return False

    # Decimal number.

    if (
        previous_text.endswith(".")
        and current_text[:1].isdigit()
    ):

        return False

    # Slash-separated survey number.

    if (
        previous_text.endswith("/")
        or current_text.startswith("/")
    ):

        return False

    # Hyphenated values.

    if (
        previous_text.endswith("-")
        or current_text.startswith("-")
    ):

        return False

    # Tamil words normally need spacing.

    return True


# ============================================================
# PREDICT TOKEN LEVEL
# ============================================================

def predict_tokens(text):

    encoded = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=MAX_LENGTH,

        return_offsets_mapping=True,

    )


    offset_mapping = encoded.pop(
        "offset_mapping"
    )


    encoded = {

        key: value.to(DEVICE)

        for key, value in encoded.items()

    }


    with torch.no_grad():

        outputs = model(
            **encoded
        )


    logits = outputs.logits[0]

    probabilities = torch.softmax(
        logits,
        dim=-1
    )


    predicted_ids = torch.argmax(
        logits,
        dim=-1
    )


    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0].detach().cpu().tolist()
    )


    results = []


    for index, token in enumerate(
        tokens
    ):

        start, end = (
            offset_mapping[0][index]
            .detach()
            .cpu()
            .tolist()
        )


        # Special token

        if start == end:

            continue


        label_id = int(
            predicted_ids[index]
            .detach()
            .cpu()
            .item()
        )


        label = id2label.get(
            label_id,
            "O"
        )


        confidence = float(
            probabilities[index][label_id]
            .detach()
            .cpu()
            .item()
        )


        original_text = text[
            start:end
        ]


        if not original_text.strip():

            continue


        results.append({

            "token":
                token,

            "text":
                original_text,

            "label":
                label,

            "confidence":
                confidence,

            "start":
                int(start),

            "end":
                int(end),

        })


    return results


# ============================================================
# MERGE BIO ENTITIES
# ============================================================

def merge_entities(token_results):

    entities = []

    current = None


    for item in token_results:

        label = item["label"]

        prefix, entity_type = split_label(
            label
        )


        # ----------------------------------------------------
        # O
        # ----------------------------------------------------

        if prefix == "O":

            if current is not None:

                entities.append(
                    current
                )

                current = None

            continue


        # ----------------------------------------------------
        # B-ENTITY
        # ----------------------------------------------------

        if prefix == "B":

            if current is not None:

                entities.append(
                    current
                )


            current = {

                "entity":
                    entity_type,

                "text":
                    item["text"],

                "confidence":
                    item["confidence"],

                "start":
                    item["start"],

                "end":
                    item["end"],

            }

            continue


        # ----------------------------------------------------
        # I-ENTITY
        # ----------------------------------------------------

        if prefix == "I":

            # Normal continuation

            if (
                current is not None
                and current["entity"]
                == entity_type
            ):

                previous_text = current[
                    "text"
                ]

                current_text = item[
                    "text"
                ]


                if should_add_space(

                    previous_text,

                    current_text,

                    current["end"],

                    item["start"],

                ):

                    current["text"] += " "


                current["text"] += (
                    current_text
                )


                current["end"] = item[
                    "end"
                ]


                current["confidence"] = (

                    current["confidence"]
                    + item["confidence"]

                ) / 2


            else:

                # Invalid I without matching B.
                # Treat it as a new entity.

                if current is not None:

                    entities.append(
                        current
                    )


                current = {

                    "entity":
                        entity_type,

                    "text":
                        item["text"],

                    "confidence":
                        item["confidence"],

                    "start":
                        item["start"],

                    "end":
                        item["end"],

                }


    if current is not None:

        entities.append(
            current
        )


    return entities


# ============================================================
# POST-PROCESS ENTITY TEXT
# ============================================================

def clean_entity_text(
    entity_type,
    text
):

    text = text.strip()


    # Remove accidental spaces around punctuation.

    text = re.sub(
        r"\s+([,.:;/%)\]])",
        r"\1",
        text
    )


    text = re.sub(
        r"([(\[/])\s+",
        r"\1",
        text
    )


    # Fix numeric values that were reconstructed
    # with unwanted spaces.

    if entity_type in [

        "PATTA_NUMBER",

        "SURVEY_NUMBER",

        "DOCUMENT_NUMBER",

        "YEAR",

        "AREA",

    ]:

        text = re.sub(
            r"(\d)\s+(\d)",
            r"\1\2",
            text
        )


        text = re.sub(
            r"(\d)\s*/\s*(\d)",
            r"\1/\2",
            text
        )


        text = re.sub(
            r"(\d)\s*\.\s*(\d)",
            r"\1.\2",
            text
        )


    return text


# ============================================================
# REMOVE OBVIOUS LOW-VALUE PREDICTIONS
# ============================================================

def filter_entities(
    entities
):

    filtered = []


    for entity in entities:

        entity_type = entity[
            "entity"
        ]

        text = clean_entity_text(

            entity_type,

            entity["text"]

        )


        confidence = entity[
            "confidence"
        ]


        # Empty

        if not text:

            continue


        # Very low confidence predictions

        if confidence < 0.50:

            continue


        # ----------------------------------------------------
        # OWNER cleanup
        # ----------------------------------------------------

        if entity_type == "OWNER":

            # Honorifics should not become owners.

            if text in [

                "திரு",

                "திருமதி",

                "திருமிகு",

                "திரு.",

                "திருமதி.",

            ]:

                continue


            if text in [
                ".",
                ",",
                ":",
                "-",
            ]:

                continue


        entity["text"] = text


        filtered.append(
            entity
        )


    return filtered


# ============================================================
# STRUCTURED RECORD
# ============================================================

def create_record(
    entities
):

    record = {

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


    key_mapping = {

        "OWNER":
            "owner",

        "FATHER_NAME":
            "father_name",

        "PATTA_NUMBER":
            "patta_number",

        "SURVEY_NUMBER":
            "survey_number",

        "AREA":
            "area",

        "BOUNDARY":
            "boundary",

        "VILLAGE":
            "village",

        "TALUK":
            "taluk",

        "DISTRICT":
            "district",

        "YEAR":
            "year",

        "DOCUMENT_NUMBER":
            "document_number",

        "LAND_TYPE":
            "land_type",

    }


    for entity in entities:

        entity_type = entity[
            "entity"
        ]


        if entity_type not in key_mapping:

            continue


        key = key_mapping[
            entity_type
        ]


        record[key].append({

            "value":
                entity["text"],

            "confidence":
                round(
                    entity["confidence"],
                    4
                ),

        })


    return record


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    text,
    token_results,
    entities,
    record
):

    print("\n")

    print("=" * 70)

    print("TOKEN PREDICTIONS")

    print("=" * 70)


    for item in token_results:

        print(

            f"{item['text']!r:<20} "
            f"{item['label']:<20} "
            f"{item['confidence'] * 100:6.2f}%"

        )


    print("\n")

    print("=" * 70)

    print("EXTRACTED LAND RECORD")

    print("=" * 70)


    if not entities:

        print(
            "\nNo entities detected."
        )

    else:

        for entity in entities:

            entity_type = entity[
                "entity"
            ]


            display_name = ENTITY_NAMES.get(

                entity_type,

                entity_type

            )


            confidence = (
                entity["confidence"]
                * 100
            )


            print(

                f"\n{display_name:<20} "
                f"→ {entity['text']} "
                f"({confidence:.1f}%)"

            )


    print("\n")

    print("=" * 70)

    print("STRUCTURED JSON")

    print("=" * 70)


    print(

        json.dumps(

            record,

            ensure_ascii=False,

            indent=2,

        )

    )


# ============================================================
# RUN ONE PREDICTION
# ============================================================

def analyze(text):

    token_results = predict_tokens(
        text
    )


    entities = merge_entities(
        token_results
    )


    entities = filter_entities(
        entities
    )


    record = create_record(
        entities
    )


    print_results(

        text,

        token_results,

        entities,

        record

    )


# ============================================================
# MAIN
# ============================================================

print("\n")

print("=" * 70)

print("       TAMIL LAND RECORD NER")

print("=" * 70)

print()

print(
    "Enter Tamil land-record text."
)

print(
    "Type 'exit' to quit."
)

print()


while True:

    try:

        text = input(
            "\nTamil text > "
        )

    except (
        KeyboardInterrupt,
        EOFError
    ):

        print(
            "\nExiting..."
        )

        break


    text = text.strip()


    if not text:

        continue


    if text.lower() in [

        "exit",

        "quit",

        "q",

    ]:

        print(
            "\nGoodbye."
        )

        break


    try:

        analyze(
            text
        )

    except Exception as error:

        print(
            "\nPrediction error:"
        )

        print(
            error
        )