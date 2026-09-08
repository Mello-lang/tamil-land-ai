import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ============================================================
# PATHS / DEVICE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "land-ner-v2"
LABELS_FILE = MODEL_DIR / "labels.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# ENTITY DEFINITIONS
# ============================================================

ENTITY_ORDER = [
    "owner",
    "father_name",
    "patta_number",
    "survey_number",
    "area",
    "boundary",
    "village",
    "taluk",
    "district",
    "year",
    "document_number",
    "land_type",
]

ENTITY_DISPLAY = {
    "owner": "Owner",
    "father_name": "Father Name",
    "patta_number": "Patta Number",
    "survey_number": "Survey Number",
    "area": "Area",
    "boundary": "Boundary",
    "village": "Village",
    "taluk": "Taluk",
    "district": "District",
    "year": "Year",
    "document_number": "Document Number",
    "land_type": "Land Type",
}


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABELS_FILE, "r", encoding="utf-8") as f:
    labels_data = json.load(f)

if "id2label" in labels_data:
    raw_labels = labels_data["id2label"]
elif "labels" in labels_data and isinstance(labels_data["labels"], dict):
    raw_labels = labels_data["labels"]
else:
    raw_labels = labels_data

id2label = {int(k): v for k, v in raw_labels.items()}


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("       TAMIL LAND RECORD NER - INDICBERT V2")
print("=" * 70)
print()
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )

print()
print(f"Labels loaded: {len(id2label)}")
print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    use_fast=True,
)

print("Tokenizer loaded.")
print()
print("Loading trained V2 NER model...")

model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# TOKEN HELPERS
# ============================================================

SPECIAL_TOKENS = {
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
    "<s>",
    "</s>",
    "<pad>",
}


def clean_token(token):
    """Remove the WordPiece ## prefix."""
    return token[2:] if token.startswith("##") else token


def is_special(token):
    return token in SPECIAL_TOKENS


def is_punctuation_token(token):
    """
    Punctuation that should normally NOT become part of an extracted
    land-record entity.

    Internal / and decimal . are handled separately.
    """
    token = clean_token(token)

    return token in {
        ",",
        ":",
        ";",
        "!",
        "?",
        "。",
        "，",
        "：",
        "；",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        '"',
        "'",
        "“",
        "”",
        "‘",
        "’",
    }


def is_decimal_point(tokens, index):
    """
    Keep '.' only when it is between numeric pieces.

    Example:
        1 . 75 -> 1.75

    But:
        72/4B . -> 72/4B
    """
    token = clean_token(tokens[index])

    if token != ".":
        return False

    previous = ""
    next_token = ""

    for i in range(index - 1, -1, -1):
        candidate = clean_token(tokens[i])
        if candidate:
            previous = candidate
            break

    for i in range(index + 1, len(tokens)):
        candidate = clean_token(tokens[i])
        if candidate:
            next_token = candidate
            break

    return (
        bool(previous)
        and bool(next_token)
        and previous[-1].isdigit()
        and next_token[0].isdigit()
    )


def join_tokens(tokens):
    """
    Reconstruct HuggingFace WordPiece tokens into readable text.

    Examples:
        ["முருக", "##ேசன்"] -> "முருகேசன்"
        ["மண", "##ப்பாறை"] -> "மணப்பாறை"
        ["72", "/", "4", "##B"] -> "72/4B"
        ["1", ".", "75", "ஏக்கர்"] -> "1.75 ஏக்கர்"

    Trailing punctuation is deliberately ignored.
    """
    if not tokens:
        return ""

    result = ""

    for index, raw_token in enumerate(tokens):
        continuation = raw_token.startswith("##")
        token = clean_token(raw_token)

        if not token:
            continue

        # Ignore normal punctuation.
        if is_punctuation_token(token):
            continue

        # Keep a decimal point only when it is between numbers.
        if token == ".":
            if is_decimal_point(tokens, index):
                result += "."
            continue

        # Slash is meaningful inside survey numbers.
        if token == "/":
            result += "/"
            continue

        # Hyphen can be meaningful inside identifiers.
        if token == "-":
            result += "-"
            continue

        if not result:
            result = token
            continue

        # WordPiece continuation.
        if continuation:
            result += token
            continue

        # Anything immediately after slash attaches directly.
        if result.endswith("/"):
            result += token
            continue

        # Anything immediately after hyphen attaches directly.
        if result.endswith("-"):
            result += token
            continue

        # Numeric fragments belong together.
        if result[-1:].isdigit() and token[0].isdigit():
            result += token
            continue

        # If result ends in a decimal point, attach the number.
        if result.endswith(".") and token[0].isdigit():
            result += token
            continue

        result += " " + token

    return result.strip()


# ============================================================
# ENTITY VALUE CLEANING
# ============================================================

def clean_value(entity_type, value):
    """
    Final normalization of extracted entity values.

    This is post-processing only. It does NOT change the trained model.
    """

    value = value.strip()

    # Remove leading/trailing separators.
    value = re.sub(r"^[,;:]+", "", value)
    value = re.sub(r"[,;:]+$", "", value)

    # Normalize whitespace around slash.
    value = re.sub(r"\s*/\s*", "/", value)

    # Normalize decimal numbers.
    value = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", value)

    # Patta / document / year are numeric identifiers.
    if entity_type in {
        "patta_number",
        "document_number",
        "year",
    }:
        value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)

    # Survey numbers such as 72/4B.
    if entity_type == "survey_number":
        value = re.sub(r"\s*/\s*", "/", value)
        value = re.sub(r"(?<=/)\s+", "", value)
        value = re.sub(r"\s+(?=[A-Za-z])", "", value)

    # Area such as 1.75 ஏக்கர்.
    if entity_type == "area":
        value = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", value)
        value = re.sub(r"(\d+(?:\.\d+)?)\s+(?=\S)", r"\1 ", value)

    # General whitespace cleanup.
    value = re.sub(r"\s+", " ", value).strip()

    # Never return a sentence-ending period as part of an entity.
    value = value.rstrip(".,;:")

    return value


# ============================================================
# MODEL INFERENCE
# ============================================================

def extract_entities(text, show_tokens=True):
    """
    Run the V2 NER model and return structured entities.

    The important part here is that punctuation predicted as I-ENTITY
    is NOT allowed to corrupt the extracted value.

    Examples:
        72/4B.       -> 72/4B
        புத்தாநத்தம். -> புத்தாநத்தம்
        புஞ்சை.      -> புஞ்சை
    """

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    # offset_mapping is useful for tokenization diagnostics but is not
    # accepted by the model forward pass.
    encoded.pop("offset_mapping", None)

    inputs = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=-1)

    predicted_ids = torch.argmax(logits, dim=-1).tolist()

    confidences = [
        float(probabilities[i, predicted_ids[i]].detach().cpu())
        for i in range(len(predicted_ids))
    ]

    token_ids = encoded["input_ids"][0].detach().cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # --------------------------------------------------------
    # TOKEN PREDICTIONS
    # --------------------------------------------------------

    if show_tokens:
        print("=" * 70)
        print("TOKEN PREDICTIONS")
        print("=" * 70)

        for token, label_id, confidence in zip(
            tokens,
            predicted_ids,
            confidences,
        ):
            label = id2label.get(label_id, "O")

            print(
                f"{token:<25}"
                f"{label:<25}"
                f"{confidence * 100:>7.2f}%"
            )

    # --------------------------------------------------------
    # ENTITY EXTRACTION
    # --------------------------------------------------------

    result = {
        entity_type: []
        for entity_type in ENTITY_ORDER
    }

    current_type = None
    current_tokens = []
    current_confidences = []

    def flush():
        nonlocal current_type
        nonlocal current_tokens
        nonlocal current_confidences

        if not current_type or not current_tokens:
            current_type = None
            current_tokens = []
            current_confidences = []
            return

        value = clean_value(
            current_type,
            join_tokens(current_tokens),
        )

        if value:
            result[current_type].append(
                {
                    "value": value,
                    "confidence": round(
                        sum(current_confidences)
                        / len(current_confidences),
                        4,
                    ),
                }
            )

        current_type = None
        current_tokens = []
        current_confidences = []

    for token, label_id, confidence in zip(
        tokens,
        predicted_ids,
        confidences,
    ):
        label = id2label.get(label_id, "O")

        # Special tokens must never become entities even if the model
        # accidentally assigns them an entity label.
        if is_special(token):
            flush()
            continue

        # ----------------------------------------------------
        # CRITICAL FIX:
        #
        # Some predictions look like:
        #
        # 72 B-SURVEY_NUMBER
        # /  I-SURVEY_NUMBER
        # 4  I-SURVEY_NUMBER
        # ##B I-SURVEY_NUMBER
        # .  I-SURVEY_NUMBER
        #
        # The final '.' is punctuation, not part of the survey
        # number. Ignore punctuation while KEEPING the entity open.
        #
        # This also prevents:
        # "புத்தாநத்தம்." -> "புத்தாநத்தம்."
        # "புஞ்சை."      -> "புஞ்சை."
        # ----------------------------------------------------

        if is_punctuation_token(token):
            # Keep slash/hyphen logic below. Normal punctuation is
            # simply ignored without flushing the entity.
            if clean_token(token) not in {"/", "-"}:
                continue

        if "-" not in label:
            flush()
            continue

        prefix, entity_name = label.split("-", 1)
        entity_type = entity_name.lower()

        if entity_type not in result:
            flush()
            continue

        # B-ENTITY starts a new entity.
        if prefix == "B":
            flush()

            current_type = entity_type
            current_tokens = [token]
            current_confidences = [confidence]

        # I-ENTITY continues the current entity.
        elif prefix == "I":
            if current_type != entity_type:
                flush()

                current_type = entity_type
                current_tokens = [token]
                current_confidences = [confidence]
            else:
                # Ignore punctuation tokens as actual entity content,
                # but DO NOT close the entity.
                if is_punctuation_token(token):
                    continue

                current_tokens.append(token)
                current_confidences.append(confidence)

    flush()

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    for entity_type in result:
        seen = set()
        unique_items = []

        for item in result[entity_type]:
            key = item["value"]

            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        result[entity_type] = unique_items

    return result


# ============================================================
# PRINT RESULTS
# ============================================================

def print_result(result):
    print()
    print("=" * 70)
    print("EXTRACTED LAND RECORD")
    print("=" * 70)

    found = False

    for entity_type in ENTITY_ORDER:
        for item in result[entity_type]:
            found = True

            print(
                f"{ENTITY_DISPLAY[entity_type]:<20}"
                f"→ {item['value']} "
                f"({item['confidence'] * 100:.1f}%)"
            )

    if not found:
        print("No entities detected.")

    print()
    print("=" * 70)
    print("STRUCTURED JSON")
    print("=" * 70)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# INTERACTIVE CLI
# ============================================================

print("=" * 70)
print("       TAMIL LAND RECORD NER V2")
print("=" * 70)
print()
print("Enter Tamil land-record text.")
print("Type 'exit' to quit.")
print()

while True:
    try:
        text = input("Tamil text > ").strip()

        if text.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not text:
            continue

        try:
            result = extract_entities(
                text,
                show_tokens=True,
            )

            print_result(result)

        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print()
                print("CUDA out of memory.")
                print("Try a shorter input.")

            else:
                print(f"\nRuntime error: {error}")

        except Exception as error:
            print(f"\nUnexpected error: {error}")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break
