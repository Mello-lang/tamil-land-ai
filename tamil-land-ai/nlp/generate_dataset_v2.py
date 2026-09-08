import json
import random
import re
from pathlib import Path

# ============================================================
# TAMIL LAND RECORD NER - DATASET V2 GENERATOR
# ============================================================

SEED = 42
random.seed(SEED)

OUTPUT_DIR = Path("data/ner_v2")

TRAIN_SIZE = 20000
VALIDATION_SIZE = 2500
TEST_SIZE = 2500

# ------------------------------------------------------------
# Label set
# ------------------------------------------------------------

ENTITY_TYPES = [
    "AREA",
    "BOUNDARY",
    "DISTRICT",
    "DOCUMENT_NUMBER",
    "FATHER_NAME",
    "LAND_TYPE",
    "OWNER",
    "PATTA_NUMBER",
    "SURVEY_NUMBER",
    "TALUK",
    "VILLAGE",
    "YEAR",
]

LABELS = ["O"]

for entity in ENTITY_TYPES:
    LABELS.append(f"B-{entity}")
    LABELS.append(f"I-{entity}")

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# ------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------

OWNERS = [
    "ராமசாமி",
    "சுப்பிரமணியன்",
    "முருகன்",
    "கண்ணன்",
    "செல்வராஜ்",
    "சரவணன்",
    "முரளி",
    "குமார்",
    "ராஜேந்திரன்",
    "சண்முகம்",
    "வெங்கடேசன்",
    "கோபால்",
    "மோகன்",
    "சுந்தரமூர்த்தி",
    "பாலசுப்பிரமணியம்",
    "தங்கவேல்",
    "பழனிச்சாமி",
    "சின்னசாமி",
    "பெருமாள்",
    "நடராஜன்",
    "விஜயகுமார்",
    "அருண்குமார்",
    "மணிகண்டன்",
    "பிரகாஷ்",
    "சிவகுமார்",
]

FATHERS = [
    "சுப்பிரமணியன்",
    "முருகன்",
    "ராமசாமி",
    "கண்ணன்",
    "பெருமாள்",
    "சண்முகம்",
    "நடராஜன்",
    "பழனிச்சாமி",
    "கோபால்",
    "வெங்கடேசன்",
    "முத்துசாமி",
    "சின்னசாமி",
    "தங்கவேல்",
    "ராஜேந்திரன்",
    "கிருஷ்ணன்",
]

DISTRICTS = [
    "கோயம்புத்தூர்",
    "திருப்பூர்",
    "ஈரோடு",
    "சேலம்",
    "நாமக்கல்",
    "கரூர்",
    "திண்டுக்கல்",
    "மதுரை",
    "திருச்சிராப்பள்ளி",
    "தஞ்சாவூர்",
    "திருநெல்வேலி",
    "தூத்துக்குடி",
    "வேலூர்",
    "காஞ்சிபுரம்",
    "செங்கல்பட்டு",
    "திருவள்ளூர்",
    "கடலூர்",
    "விழுப்புரம்",
    "கிருஷ்ணகிரி",
    "தர்மபுரி",
]

TALUKS = [
    "பொள்ளாச்சி",
    "மேட்டுப்பாளையம்",
    "சூலூர்",
    "கிணத்துக்கடவு",
    "அவிநாசி",
    "பல்லடம்",
    "உடுமலைப்பேட்டை",
    "தாராபுரம்",
    "பவானி",
    "கோபிச்செட்டிபாளையம்",
    "சத்தியமங்கலம்",
    "ஈரோடு",
    "கரூர்",
    "அரவக்குறிச்சி",
    "திண்டுக்கல்",
    "பழனி",
    "ஒட்டன்சத்திரம்",
    "மதுரை",
]

VILLAGES = [
    "ஆனைமலை",
    "கோட்டூர்",
    "வேட்டைக்காரன்புதூர்",
    "சோமந்துறை",
    "கிணத்துக்கடவு",
    "நெகமம்",
    "சுல்தான்பேட்டை",
    "காளப்பட்டி",
    "பேரூர்",
    "சரவணம்பட்டி",
    "பொள்ளாச்சி",
    "மரப்பாலம்",
    "குறிச்சி",
    "சின்னவேடம்பட்டி",
    "வடுகபாளையம்",
    "செட்டிபாளையம்",
    "கருமத்தம்பட்டி",
    "மதுக்கரை",
    "சூலூர்",
    "துடியலூர்",
]

LAND_TYPES = [
    "நஞ்சை",
    "புஞ்சை",
    "நன்செய்",
    "புன்செய்",
    "விவசாய நிலம்",
    "வீட்டு மனை",
    "மனை",
    "தோட்ட நிலம்",
    "விவசாயம்",
]

YEARS = list(range(1950, 2027))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def random_survey():
    a = random.randint(1, 999)
    b = random.randint(1, 99)

    styles = [
        f"{a}/{b}",
        f"{a}-{b}",
        f"{a}/{b}A",
        f"{a}/{b}B",
        f"{a}",
    ]

    return random.choice(styles)


def random_patta():
    return str(random.randint(100, 999999))


def random_document():
    return str(random.randint(100, 999999))


def random_area():
    value = random.choice([
        round(random.uniform(0.10, 0.99), 2),
        round(random.uniform(1.00, 9.99), 2),
        round(random.uniform(10.00, 99.99), 2),
        random.randint(1, 50),
    ])

    unit = random.choice([
        "ஏக்கர்",
        "ஏக்கர்கள்",
        "சென்ட்",
        "சென்ட் நிலம்",
    ])

    return f"{value:g} {unit}"


def make_entity(text, entity_type):
    return {
        "text": text,
        "label": entity_type,
    }


def tokenize_with_entities(parts):
    """
    parts:
        [
            ("உரிமையாளர் ", None),
            ("ராமசாமி", "OWNER"),
            (" தந்தை ", None),
            ("சுப்பிரமணியன்", "FATHER_NAME")
        ]
    """

    text = ""
    entities = []

    for part, label in parts:
        start = len(text)
        text += part
        end = len(text)

        if label:
            entities.append({
                "start": start,
                "end": end,
                "label": label,
            })

    # Character-offset entities -> word labels
    words = re.findall(r"\S+", text)

    tokens = []
    labels = []

    cursor = 0

    for word in words:
        start = text.find(word, cursor)
        end = start + len(word)
        cursor = end

        token_label = "O"

        overlapping = []

        for ent in entities:
            if ent["start"] < end and ent["end"] > start:
                overlapping.append(ent)

        if overlapping:
            ent = overlapping[0]

            if start <= ent["start"]:
                token_label = "B-" + ent["label"]
            else:
                token_label = "I-" + ent["label"]

        tokens.append(word)
        labels.append(token_label)

    return {
        "tokens": tokens,
        "labels": labels,
    }


# ------------------------------------------------------------
# Template generators
# ------------------------------------------------------------

def template_owner_father():
    owner = random.choice(OWNERS)
    father = random.choice(FATHERS)

    style = random.randint(1, 8)

    if style == 1:
        return tokenize_with_entities([
            ("உரிமையாளர் ", None),
            (owner, "OWNER"),
            (", தந்தை பெயர் ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ])

    if style == 2:
        return tokenize_with_entities([
            (owner, "OWNER"),
            (" த/பெ ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ])

    if style == 3:
        return tokenize_with_entities([
            ("திரு. ", None),
            (owner, "OWNER"),
            (", தந்தை ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ])

    if style == 4:
        return tokenize_with_entities([
            ("பெயர்: ", None),
            (owner, "OWNER"),
            (", தந்தையின் பெயர்: ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ])

    if style == 5:
        return tokenize_with_entities([
            ("பட்டாதாரர் ", None),
            (owner, "OWNER"),
            (", த/பெ ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ])

    if style == 6:
        return tokenize_with_entities([
            ("திரு ", None),
            (owner, "OWNER"),
            (" அவர்களின் தந்தை ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ])

    if style == 7:
        return tokenize_with_entities([
            (owner, "OWNER"),
            (" என்பவரின் தந்தை பெயர் ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ])

    return tokenize_with_entities([
        ("விண்ணப்பதாரர் ", None),
        (owner, "OWNER"),
        (", தந்தை பெயர்: ", None),
        (father, "FATHER_NAME"),
        (".", None),
    ])


def template_location():
    district = random.choice(DISTRICTS)
    taluk = random.choice(TALUKS)
    village = random.choice(VILLAGES)

    style = random.randint(1, 7)

    if style == 1:
        return tokenize_with_entities([
            (district, "DISTRICT"),
            (" மாவட்டம், ", None),
            (taluk, "TALUK"),
            (" வட்டம், ", None),
            (village, "VILLAGE"),
            (" கிராமம்.", None),
        ])

    if style == 2:
        return tokenize_with_entities([
            ("மாவட்டம்: ", None),
            (district, "DISTRICT"),
            (", வட்டம்: ", None),
            (taluk, "TALUK"),
            (", கிராமம்: ", None),
            (village, "VILLAGE"),
            (".", None),
        ])

    if style == 3:
        return tokenize_with_entities([
            (village, "VILLAGE"),
            (" கிராமம், ", None),
            (taluk, "TALUK"),
            (" தாலுகா, ", None),
            (district, "DISTRICT"),
            (" மாவட்டம்.", None),
        ])

    if style == 4:
        return tokenize_with_entities([
            ("ஊர் ", None),
            (village, "VILLAGE"),
            (", தாலுகா ", None),
            (taluk, "TALUK"),
            (", மாவட்டம் ", None),
            (district, "DISTRICT"),
            (".", None),
        ])

    if style == 5:
        return tokenize_with_entities([
            (district, "DISTRICT"),
            (" மாவட்டத்தில் உள்ள ", None),
            (taluk, "TALUK"),
            (" வட்டத்தின் ", None),
            (village, "VILLAGE"),
            (" கிராமம்.", None),
        ])

    if style == 6:
        return tokenize_with_entities([
            ("கிராமம் ", None),
            (village, "VILLAGE"),
            (", வட்டம் ", None),
            (taluk, "TALUK"),
            (", மாவட்டம் ", None),
            (district, "DISTRICT"),
            (".", None),
        ])

    return tokenize_with_entities([
        (village, "VILLAGE"),
        (" கிராமம் அமைந்துள்ள ", None),
        (taluk, "TALUK"),
        (" வட்டம், ", None),
        (district, "DISTRICT"),
        (" மாவட்டம்.", None),
    ])


def template_survey_patta():
    survey = random_survey()
    patta = random_patta()

    style = random.randint(1, 7)

    if style == 1:
        return tokenize_with_entities([
            ("சர்வே எண் ", None),
            (survey, "SURVEY_NUMBER"),
            (", பட்டா எண் ", None),
            (patta, "PATTA_NUMBER"),
            (".", None),
        ])

    if style == 2:
        return tokenize_with_entities([
            ("பட்டா எண் ", None),
            (patta, "PATTA_NUMBER"),
            (" மற்றும் சர்வே எண் ", None),
            (survey, "SURVEY_NUMBER"),
            (".", None),
        ])

    if style == 3:
        return tokenize_with_entities([
            ("ச.ந ", None),
            (survey, "SURVEY_NUMBER"),
            (", ப.ந ", None),
            (patta, "PATTA_NUMBER"),
            (".", None),
        ])

    if style == 4:
        return tokenize_with_entities([
            ("சர்வே நம்பர்: ", None),
            (survey, "SURVEY_NUMBER"),
            (". பட்டா நம்பர்: ", None),
            (patta, "PATTA_NUMBER"),
            (".", None),
        ])

    if style == 5:
        return tokenize_with_entities([
            ("Survey No. ", None),
            (survey, "SURVEY_NUMBER"),
            (", Patta No. ", None),
            (patta, "PATTA_NUMBER"),
            (".", None),
        ])

    if style == 6:
        return tokenize_with_entities([
            ("பட்டா ", None),
            (patta, "PATTA_NUMBER"),
            (" உடைய நிலத்தின் சர்வே எண் ", None),
            (survey, "SURVEY_NUMBER"),
            (".", None),
        ])

    return tokenize_with_entities([
        ("நிலம் சர்வே எண் ", None),
        (survey, "SURVEY_NUMBER"),
        (" உட்பட்டது. பட்டா எண் ", None),
        (patta, "PATTA_NUMBER"),
        (".", None),
    ])


def template_area_land():
    area = random_area()
    land_type = random.choice(LAND_TYPES)

    style = random.randint(1, 6)

    if style == 1:
        return tokenize_with_entities([
            (area, "AREA"),
            (" ", None),
            (land_type, "LAND_TYPE"),
            (" நிலம்.", None),
        ])

    if style == 2:
        return tokenize_with_entities([
            (land_type, "LAND_TYPE"),
            (" நிலம் ", None),
            (area, "AREA"),
            (" அளவில் உள்ளது.", None),
        ])

    if style == 3:
        return tokenize_with_entities([
            ("மொத்த பரப்பளவு ", None),
            (area, "AREA"),
            (", நில வகை ", None),
            (land_type, "LAND_TYPE"),
            (".", None),
        ])

    if style == 4:
        return tokenize_with_entities([
            ("நிலத்தின் பரப்பு ", None),
            (area, "AREA"),
            (". இது ", None),
            (land_type, "LAND_TYPE"),
            (" நிலமாகும்.", None),
        ])

    if style == 5:
        return tokenize_with_entities([
            (area, "AREA"),
            (" கொண்ட ", None),
            (land_type, "LAND_TYPE"),
            (" வகை நிலம்.", None),
        ])

    return tokenize_with_entities([
        ("பரப்பு: ", None),
        (area, "AREA"),
        (", வகை: ", None),
        (land_type, "LAND_TYPE"),
        (".", None),
    ])


def template_document_year():
    document = random_document()
    year = random.choice(YEARS)

    style = random.randint(1, 5)

    if style == 1:
        return tokenize_with_entities([
            ("ஆவண எண் ", None),
            (document, "DOCUMENT_NUMBER"),
            (", ஆண்டு ", None),
            (str(year), "YEAR"),
            (".", None),
        ])

    if style == 2:
        return tokenize_with_entities([
            ("ஆண்டு ", None),
            (str(year), "YEAR"),
            (" ஆம் ஆண்டின் ஆவண எண் ", None),
            (document, "DOCUMENT_NUMBER"),
            (".", None),
        ])

    if style == 3:
        return tokenize_with_entities([
            ("Doc No: ", None),
            (document, "DOCUMENT_NUMBER"),
            (", Year: ", None),
            (str(year), "YEAR"),
            (".", None),
        ])

    if style == 4:
        return tokenize_with_entities([
            ("பதிவு ஆண்டு ", None),
            (str(year), "YEAR"),
            (". ஆவண பதிவு எண் ", None),
            (document, "DOCUMENT_NUMBER"),
            (".", None),
        ])

    return tokenize_with_entities([
        ("ஆவண பதிவு எண்: ", None),
        (document, "DOCUMENT_NUMBER"),
        ("; பதிவு ஆண்டு: ", None),
        (str(year), "YEAR"),
        (".", None),
    ])


def template_boundary():
    north = random.choice(VILLAGES)
    south = random.choice(VILLAGES)
    east = random.choice(VILLAGES)
    west = random.choice(VILLAGES)

    return tokenize_with_entities([
        ("வடக்கு: ", None),
        (north, "BOUNDARY"),
        (", தெற்கு: ", None),
        (south, "BOUNDARY"),
        (", கிழக்கு: ", None),
        (east, "BOUNDARY"),
        (", மேற்கு: ", None),
        (west, "BOUNDARY"),
        (".", None),
    ])


# ------------------------------------------------------------
# Combined realistic records
# ------------------------------------------------------------

def template_full_record():
    owner = random.choice(OWNERS)
    father = random.choice(FATHERS)

    district = random.choice(DISTRICTS)
    taluk = random.choice(TALUKS)
    village = random.choice(VILLAGES)

    survey = random_survey()
    patta = random_patta()

    area = random_area()
    land_type = random.choice(LAND_TYPES)

    document = random_document()
    year = random.choice(YEARS)

    style = random.randint(1, 10)

    if style == 1:
        parts = [
            ("உரிமையாளர் ", None),
            (owner, "OWNER"),
            (", தந்தை பெயர் ", None),
            (father, "FATHER_NAME"),
            (", சர்வே எண் ", None),
            (survey, "SURVEY_NUMBER"),
            (", பட்டா எண் ", None),
            (patta, "PATTA_NUMBER"),
            (", ", None),
            (village, "VILLAGE"),
            (" கிராமம், ", None),
            (taluk, "TALUK"),
            (" வட்டம், ", None),
            (district, "DISTRICT"),
            (" மாவட்டம், ", None),
            (area, "AREA"),
            (" ", None),
            (land_type, "LAND_TYPE"),
            (" நிலம்.", None),
        ]

    elif style == 2:
        parts = [
            (district, "DISTRICT"),
            (" மாவட்டம், ", None),
            (taluk, "TALUK"),
            (" தாலுகா, ", None),
            (village, "VILLAGE"),
            (" கிராமத்தில் ", None),
            (survey, "SURVEY_NUMBER"),
            (" என்ற சர்வே எண்ணில் ", None),
            (area, "AREA"),
            (" ", None),
            (land_type, "LAND_TYPE"),
            (" நிலம். பட்டாதாரர் ", None),
            (owner, "OWNER"),
            (".", None),
        ]

    elif style == 3:
        parts = [
            ("பட்டா எண் ", None),
            (patta, "PATTA_NUMBER"),
            (" உடைய ", None),
            (owner, "OWNER"),
            (" அவர்களின் தந்தை ", None),
            (father, "FATHER_NAME"),
            (". சர்வே ", None),
            (survey, "SURVEY_NUMBER"),
            (", ", None),
            (village, "VILLAGE"),
            (" கிராமம், ", None),
            (taluk, "TALUK"),
            (" வட்டம், ", None),
            (district, "DISTRICT"),
            (" மாவட்டம்.", None),
        ]

    elif style == 4:
        parts = [
            ("திரு. ", None),
            (owner, "OWNER"),
            (" த/பெ ", None),
            (father, "FATHER_NAME"),
            (" என்பவருக்கு சொந்தமான ", None),
            (area, "AREA"),
            (" ", None),
            (land_type, "LAND_TYPE"),
            (" நிலம். சர்வே எண் ", None),
            (survey, "SURVEY_NUMBER"),
            (", பட்டா எண் ", None),
            (patta, "PATTA_NUMBER"),
            (".", None),
        ]

    elif style == 5:
        parts = [
            ("ஆவண எண் ", None),
            (document, "DOCUMENT_NUMBER"),
            (" / ", None),
            (str(year), "YEAR"),
            (" ஆண்டு. உரிமையாளர் ", None),
            (owner, "OWNER"),
            (". சர்வே எண் ", None),
            (survey, "SURVEY_NUMBER"),
            (".", None),
        ]

    elif style == 6:
        parts = [
            ("கிராமம்: ", None),
            (village, "VILLAGE"),
            (", வட்டம்: ", None),
            (taluk, "TALUK"),
            (", மாவட்டம்: ", None),
            (district, "DISTRICT"),
            (". உரிமையாளர்: ", None),
            (owner, "OWNER"),
            (". தந்தை: ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ]

    elif style == 7:
        parts = [
            ("நில வகை ", None),
            (land_type, "LAND_TYPE"),
            (". பரப்பு ", None),
            (area, "AREA"),
            (". சர்வே எண் ", None),
            (survey, "SURVEY_NUMBER"),
            (". பட்டா எண் ", None),
            (patta, "PATTA_NUMBER"),
            (".", None),
        ]

    elif style == 8:
        parts = [
            ("பட்டாதாரர் பெயர் ", None),
            (owner, "OWNER"),
            (", த/பெ ", None),
            (father, "FATHER_NAME"),
            (". ", None),
            (district, "DISTRICT"),
            (" மாவட்டம், ", None),
            (taluk, "TALUK"),
            (" வட்டம், ", None),
            (village, "VILLAGE"),
            (" கிராமம்.", None),
        ]

    elif style == 9:
        parts = [
            ("சர்வே எண் ", None),
            (survey, "SURVEY_NUMBER"),
            (" இல் ", None),
            (area, "AREA"),
            (" பரப்புள்ள ", None),
            (land_type, "LAND_TYPE"),
            (" நிலம் ", None),
            (owner, "OWNER"),
            (" என்பவருக்கு சொந்தமானது.", None),
        ]

    else:
        parts = [
            ("பதிவு ஆண்டு ", None),
            (str(year), "YEAR"),
            (", ஆவண எண் ", None),
            (document, "DOCUMENT_NUMBER"),
            (". ", None),
            ("பட்டா ", None),
            (patta, "PATTA_NUMBER"),
            (", சர்வே ", None),
            (survey, "SURVEY_NUMBER"),
            (". உரிமையாளர் ", None),
            (owner, "OWNER"),
            ("; தந்தை ", None),
            (father, "FATHER_NAME"),
            (".", None),
        ]

    return tokenize_with_entities(parts)


# ------------------------------------------------------------
# Generate examples
# ------------------------------------------------------------

TEMPLATES = [
    template_owner_father,
    template_location,
    template_survey_patta,
    template_area_land,
    template_document_year,
    template_boundary,
    template_full_record,
]


def generate_example():
    template = random.choice(TEMPLATES)
    return template()


def generate_dataset(size):
    examples = []

    for _ in range(size):
        examples.append(generate_example())

    return examples


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate_example(example):
    assert "tokens" in example
    assert "labels" in example

    assert len(example["tokens"]) == len(example["labels"])

    for label in example["labels"]:
        assert label in LABEL2ID, f"Unknown label: {label}"

    return True


def validate_dataset(dataset):
    for example in dataset:
        validate_example(example)

    return True


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("       TAMIL LAND RECORD NER - DATASET V2")
    print("=" * 70)

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR.resolve()}")

    print()
    print("Generating datasets...")
    print(f"  Training:    {TRAIN_SIZE}")
    print(f"  Validation:  {VALIDATION_SIZE}")
    print(f"  Test:        {TEST_SIZE}")

    train = generate_dataset(TRAIN_SIZE)
    validation = generate_dataset(VALIDATION_SIZE)
    test = generate_dataset(TEST_SIZE)

    print()
    print("Validating datasets...")

    validate_dataset(train)
    validate_dataset(validation)
    validate_dataset(test)

    print("Dataset validation: OK")

    save_json(
        OUTPUT_DIR / "train.json",
        train,
    )

    save_json(
        OUTPUT_DIR / "validation.json",
        validation,
    )

    save_json(
        OUTPUT_DIR / "test.json",
        test,
    )

    save_json(
        OUTPUT_DIR / "labels.json",
        {
            "label2id": LABEL2ID,
            "id2label": {
                str(k): v
                for k, v in ID2LABEL.items()
            },
        },
    )

    print()
    print("=" * 70)
    print("DATASET V2 GENERATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Files:")

    print(f"  {OUTPUT_DIR / 'train.json'}")
    print(f"  {OUTPUT_DIR / 'validation.json'}")
    print(f"  {OUTPUT_DIR / 'test.json'}")
    print(f"  {OUTPUT_DIR / 'labels.json'}")

    print()
    print("Examples:")
    print("-" * 70)

    for i in range(5):
        example = random.choice(train)

        print()
        print("TEXT:")
        print(" ".join(example["tokens"]))

        print()
        print("LABELS:")

        for token, label in zip(
            example["tokens"],
            example["labels"],
        ):
            if label != "O":
                print(f"  {token:<25} {label}")

        print("-" * 70)

    print()
    print("Next step:")
    print()
    print("Train IndicBERT using:")
    print()
    print("  python nlp\\train_ner_v2.py")
    print()


if __name__ == "__main__":
    main()