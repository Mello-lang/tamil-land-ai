"""
======================================================================
        TAMIL LAND RECORD NER - DATASET V3
======================================================================

Purpose:
    Generate a realistic Tamil land-record NER dataset for IndicBERT.

V3 improvements over V2:
    - Longer paragraphs
    - Multiple owners
    - Multiple survey numbers
    - Multiple patta numbers
    - Multiple land parcels
    - Document numbers
    - Years
    - Boundary descriptions
    - Different field ordering
    - Different Tamil wording
    - Multi-word entities
    - Decimal areas
    - Acre / cent / square-feet area formats
    - Subdivision survey numbers such as 72/4B, 124/3A1
    - Long mixed records
    - Hard negative numeric patterns
    - More realistic punctuation
    - Stronger boundary examples

Output:
    data/ner_v3/train.json
    data/ner_v3/validation.json
    data/ner_v3/test.json
    data/ner_v3/labels.json

Usage:
    python nlp/generate_dataset_v3.py
"""

import json
import os
import random
import re
from pathlib import Path


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 20260908

TRAIN_SIZE = 30000
VALIDATION_SIZE = 4000
TEST_SIZE = 4000

OUTPUT_DIR = Path("data") / "ner_v3"

random.seed(SEED)


# ======================================================================
# LABELS
# ======================================================================

LABELS = [
    "O",

    "B-AREA",
    "I-AREA",

    "B-BOUNDARY",
    "I-BOUNDARY",

    "B-DISTRICT",
    "I-DISTRICT",

    "B-DOCUMENT_NUMBER",
    "I-DOCUMENT_NUMBER",

    "B-FATHER_NAME",
    "I-FATHER_NAME",

    "B-LAND_TYPE",
    "I-LAND_TYPE",

    "B-OWNER",
    "I-OWNER",

    "B-PATTA_NUMBER",
    "I-PATTA_NUMBER",

    "B-SURVEY_NUMBER",
    "I-SURVEY_NUMBER",

    "B-TALUK",
    "I-TALUK",

    "B-VILLAGE",
    "I-VILLAGE",

    "B-YEAR",
    "I-YEAR",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}


# ======================================================================
# DATA VOCABULARY
# ======================================================================

FIRST_NAMES = [
    "ராமசாமி",
    "முருகேசன்",
    "செந்தில்குமார்",
    "சுப்பிரமணியன்",
    "பழனிச்சாமி",
    "கண்ணன்",
    "கோபால்",
    "சிவக்குமார்",
    "மணிகண்டன்",
    "ராஜேந்திரன்",
    "வெங்கடேசன்",
    "சரவணன்",
    "மோகன்",
    "குமார்",
    "சுரேஷ்",
    "மகேந்திரன்",
    "பாலசுப்பிரமணியன்",
    "அருணாசலம்",
    "துரைசாமி",
    "நடராஜன்",
    "ஜெயக்குமார்",
    "பிரபாகரன்",
    "விஜயகுமார்",
    "தனபால்",
    "சண்முகம்",
    "கிருஷ்ணமூர்த்தி",
    "ரமேஷ்",
    "அன்பழகன்",
    "சக்திவேல்",
    "கார்த்திகேயன்",
]

VILLAGES = [
    "ஆனைமலை",
    "காளப்பட்டி",
    "சோமந்துறை",
    "புத்தாநத்தம்",
    "செட்டிபாளையம்",
    "கிணத்துக்கடவு",
    "பொள்ளாச்சி",
    "மதுக்கரை",
    "சுல்தான்பேட்டை",
    "கோவில்பாளையம்",
    "வடக்கலூர்",
    "வெள்ளலூர்",
    "சரவணம்பட்டி",
    "குனியமுத்தூர்",
    "பேரூர்",
    "நாச்சிபாளையம்",
    "வால்பாறை",
    "உடுமலை",
    "கிணத்துக்கடவு",
    "குரும்பபாளையம்",
    "சின்னவேடம்பட்டி",
    "துடியலூர்",
    "பெரியநாயக்கன்பாளையம்",
    "அன்னூர்",
    "மேட்டுப்பாளையம்",
    "காரமடை",
    "சிங்காநல்லூர்",
    "இடையர்பாளையம்",
    "சூலூர்",
    "ஒத்தக்கால் மண்டபம்",
]

TALUKS = [
    "பொள்ளாச்சி",
    "மணப்பாறை",
    "கோவை தெற்கு",
    "கோவை வடக்கு",
    "மேட்டுப்பாளையம்",
    "சூலூர்",
    "உடுமலைப்பேட்டை",
    "வால்பாறை",
    "திருப்பூர்",
    "பல்லடம்",
    "அவிநாசி",
    "பழனி",
    "திண்டுக்கல்",
    "கரூர்",
    "நாமக்கல்",
    "சேலம்",
    "ஈரோடு",
    "கோபிசெட்டிபாளையம்",
    "திருச்செங்கோடு",
    "மதுரை வடக்கு",
    "மதுரை தெற்கு",
]

DISTRICTS = [
    "கோயம்புத்தூர்",
    "திருச்சிராப்பள்ளி",
    "மதுரை",
    "திண்டுக்கல்",
    "திருப்பூர்",
    "ஈரோடு",
    "சேலம்",
    "கரூர்",
    "நாமக்கல்",
    "திருநெல்வேலி",
    "தூத்துக்குடி",
    "விருதுநகர்",
    "தேனி",
    "கன்னியாகுமரி",
    "தஞ்சாவூர்",
    "திருவாரூர்",
    "நாகப்பட்டினம்",
    "கடலூர்",
    "விழுப்புரம்",
    "காஞ்சிபுரம்",
    "செங்கல்பட்டு",
    "திருவள்ளூர்",
    "வேலூர்",
    "ராணிப்பேட்டை",
    "திருப்பத்தூர்",
]

LAND_TYPES = [
    "நஞ்சை",
    "புஞ்சை",
    "விவசாய நிலம்",
    "வீட்டு மனை",
    "தோட்ட நிலம்",
    "மானாவாரி நிலம்",
    "விவசாய மனை",
]

BOUNDARY_PLACES = [
    "காளப்பட்டி",
    "சோமந்துறை",
    "ஆனைமலை",
    "பேரூர் சாலை",
    "ஆழியாறு கால்வாய்",
    "பொள்ளாச்சி சாலை",
    "முக்கிய சாலை",
    "கிராம சாலை",
    "வாய்க்கால்",
    "கால்வாய்",
    "அரசு சாலை",
    "தனியார் நிலம்",
    "முத்துசாமி நிலம்",
    "ராமசாமி நிலம்",
    "பொது பாதை",
    "கிழக்கு பிரதான சாலை",
    "மேற்கு கால்வாய்",
]

CONNECTORS = [
    "மற்றும்",
    "ஆகிய",
    "மேலும்",
    "உள்ள",
    "அமைந்துள்ள",
    "சேர்ந்த",
]


# ======================================================================
# BASIC HELPERS
# ======================================================================

def random_name(exclude=None):
    values = [x for x in FIRST_NAMES if x != exclude]

    if not values:
        values = FIRST_NAMES

    return random.choice(values)


def random_village():
    return random.choice(VILLAGES)


def random_taluk():
    return random.choice(TALUKS)


def random_district():
    return random.choice(DISTRICTS)


def random_year():
    return random.randint(1950, 2026)


def random_document_number():
    return f"{random.randint(100000, 999999)}"


def random_patta_number():
    return f"{random.randint(100000, 999999)}"


def random_survey_number():
    main = random.randint(1, 999)

    formats = [
        f"{main}/{random.randint(1, 20)}",
        f"{main}/{random.randint(1, 20)}A",
        f"{main}/{random.randint(1, 20)}B",
        f"{main}/{random.randint(1, 20)}C",
        f"{main}/{random.randint(1, 20)}A1",
        f"{main}/{random.randint(1, 20)}B1",
        f"{main}/{random.randint(1, 20)}A2",
        f"{main}/{random.randint(1, 20)}-1",
    ]

    return random.choice(formats)


def random_area():
    formats = [
        f"{random.randint(1, 9)}.{random.randint(1, 99):02d} ஏக்கர்",
        f"{random.randint(1, 9)}.{random.randint(1, 9)} ஏக்கர்",
        f"{random.randint(1, 20)} ஏக்கர்",
        f"{random.randint(10, 90)} சென்ட்",
        f"{random.randint(1, 9)} ஏக்கர் {random.randint(1, 99)} சென்ட்",
        f"{random.randint(500, 9000)} சதுர அடி",
        f"{random.randint(1000, 20000)} சதுர அடி",
    ]

    return random.choice(formats)


def random_simple_area():
    return random.choice([
        f"{random.randint(1, 9)}.{random.randint(1, 99):02d} ஏக்கர்",
        f"{random.randint(1, 9)}.{random.randint(1, 9)} ஏக்கர்",
        f"{random.randint(10, 90)} சென்ட்",
    ])


# ======================================================================
# TOKEN BUILDING
# ======================================================================

def clean_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def literal_tokens(text):
    """
    Convert ordinary text into tokens.

    Punctuation is kept as separate tokens so that entities don't
    accidentally absorb punctuation.
    """

    text = clean_spaces(text)

    if not text:
        return []

    return re.findall(
        r"[^\s,;:.!?()\[\]{}]+|[,;:.!?()\[\]{}]",
        text
    )


def add_literal(tokens, labels, text):
    for token in literal_tokens(text):
        tokens.append(token)
        labels.append("O")


def add_entity(tokens, labels, text, entity_type):
    """
    Add an entity using BIO labels.

    Multi-word entities become:
        B-TYPE
        I-TYPE
    """

    parts = literal_tokens(text)

    if not parts:
        return

    for i, token in enumerate(parts):
        prefix = "B-" if i == 0 else "I-"
        tokens.append(token)
        labels.append(prefix + entity_type)


def add_punctuation(tokens, labels, punctuation):
    tokens.append(punctuation)
    labels.append("O")


# ======================================================================
# RECORD BUILDING
# ======================================================================

def new_record():
    return {
        "tokens": [],
        "labels": [],
    }


def add_owner(record, owner, father=None, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    if style == 0:
        add_literal(tokens, labels, "உரிமையாளர்")
        add_entity(tokens, labels, owner, "OWNER")
        add_punctuation(tokens, labels, ",")

        if father:
            add_literal(tokens, labels, "தந்தை பெயர்")
            add_entity(tokens, labels, father, "FATHER_NAME")
            add_punctuation(tokens, labels, ",")

    elif style == 1:
        add_entity(tokens, labels, owner, "OWNER")
        add_literal(tokens, labels, "த/பெ")
        add_entity(tokens, labels, father, "FATHER_NAME")
        add_punctuation(tokens, labels, ",")

    elif style == 2:
        add_literal(tokens, labels, "திரு.")
        add_entity(tokens, labels, owner, "OWNER")
        add_literal(tokens, labels, "என்பவர்")
        add_punctuation(tokens, labels, ",")

        if father:
            add_literal(tokens, labels, "தந்தை")
            add_entity(tokens, labels, father, "FATHER_NAME")
            add_punctuation(tokens, labels, ",")

    elif style == 3:
        add_literal(tokens, labels, "நிலத்தின் உரிமையாளர்")
        add_entity(tokens, labels, owner, "OWNER")
        add_literal(tokens, labels, "தந்தை")
        add_entity(tokens, labels, father, "FATHER_NAME")
        add_punctuation(tokens, labels, ",")


def add_location(record, village, taluk, district, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    if style == 0:
        add_entity(tokens, labels, district, "DISTRICT")
        add_literal(tokens, labels, "மாவட்டம்")
        add_punctuation(tokens, labels, ",")

        add_entity(tokens, labels, taluk, "TALUK")
        add_literal(tokens, labels, "வட்டம்")
        add_punctuation(tokens, labels, ",")

        add_entity(tokens, labels, village, "VILLAGE")
        add_literal(tokens, labels, "கிராமம்")
        add_punctuation(tokens, labels, ",")

    elif style == 1:
        add_literal(tokens, labels, "மாவட்டம்")
        add_entity(tokens, labels, district, "DISTRICT")
        add_punctuation(tokens, labels, ";")

        add_literal(tokens, labels, "வட்டம்")
        add_entity(tokens, labels, taluk, "TALUK")
        add_punctuation(tokens, labels, ";")

        add_literal(tokens, labels, "கிராமம்")
        add_entity(tokens, labels, village, "VILLAGE")
        add_punctuation(tokens, labels, ",")

    elif style == 2:
        add_entity(tokens, labels, village, "VILLAGE")
        add_literal(tokens, labels, "கிராமம் அமைந்துள்ள")
        add_entity(tokens, labels, taluk, "TALUK")
        add_literal(tokens, labels, "வட்டம்")
        add_entity(tokens, labels, district, "DISTRICT")
        add_literal(tokens, labels, "மாவட்டம்")
        add_punctuation(tokens, labels, ",")

    elif style == 3:
        add_literal(tokens, labels, "இடம்")
        add_entity(tokens, labels, village, "VILLAGE")
        add_literal(tokens, labels, ",")
        add_entity(tokens, labels, taluk, "TALUK")
        add_literal(tokens, labels, "வட்டம்")
        add_entity(tokens, labels, district, "DISTRICT")
        add_literal(tokens, labels, "மாவட்டம்")
        add_punctuation(tokens, labels, ",")


def add_survey(record, survey, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    if style == 0:
        add_literal(tokens, labels, "சர்வே எண்")
        add_entity(tokens, labels, survey, "SURVEY_NUMBER")
        add_punctuation(tokens, labels, ",")

    elif style == 1:
        add_literal(tokens, labels, "சர்வே")
        add_entity(tokens, labels, survey, "SURVEY_NUMBER")
        add_punctuation(tokens, labels, ",")

    elif style == 2:
        add_literal(tokens, labels, "சர்வே எண்:")
        add_entity(tokens, labels, survey, "SURVEY_NUMBER")
        add_punctuation(tokens, labels, ",")

    elif style == 3:
        add_literal(tokens, labels, "சர்வே எண்ணான")
        add_entity(tokens, labels, survey, "SURVEY_NUMBER")
        add_punctuation(tokens, labels, ",")


def add_patta(record, patta, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    if style == 0:
        add_literal(tokens, labels, "பட்டா எண்")
        add_entity(tokens, labels, patta, "PATTA_NUMBER")
        add_punctuation(tokens, labels, ",")

    elif style == 1:
        add_literal(tokens, labels, "பட்டா")
        add_entity(tokens, labels, patta, "PATTA_NUMBER")
        add_punctuation(tokens, labels, ",")

    elif style == 2:
        add_literal(tokens, labels, "பட்டா எண்:")
        add_entity(tokens, labels, patta, "PATTA_NUMBER")
        add_punctuation(tokens, labels, ",")

    elif style == 3:
        add_literal(tokens, labels, "பட்டா எண்ணாக")
        add_entity(tokens, labels, patta, "PATTA_NUMBER")
        add_punctuation(tokens, labels, ",")


def add_area(record, area, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    if style == 0:
        add_literal(tokens, labels, "பரப்பளவு")
        add_entity(tokens, labels, area, "AREA")
        add_punctuation(tokens, labels, ",")

    elif style == 1:
        add_entity(tokens, labels, area, "AREA")
        add_literal(tokens, labels, "பரப்பளவு")
        add_punctuation(tokens, labels, ",")

    elif style == 2:
        add_literal(tokens, labels, "மொத்த பரப்பு")
        add_entity(tokens, labels, area, "AREA")
        add_punctuation(tokens, labels, ",")

    elif style == 3:
        add_literal(tokens, labels, "நிலத்தின் பரப்பளவு")
        add_entity(tokens, labels, area, "AREA")
        add_punctuation(tokens, labels, ",")


def add_land_type(record, land_type, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    if style == 0:
        add_literal(tokens, labels, "வகை")
        add_entity(tokens, labels, land_type, "LAND_TYPE")
        add_literal(tokens, labels, "நிலம்")
        add_punctuation(tokens, labels, ".")

    elif style == 1:
        add_entity(tokens, labels, land_type, "LAND_TYPE")
        add_literal(tokens, labels, "நிலமாகும்")
        add_punctuation(tokens, labels, ".")

    elif style == 2:
        add_literal(tokens, labels, "நில வகை")
        add_entity(tokens, labels, land_type, "LAND_TYPE")
        add_punctuation(tokens, labels, ".")

    elif style == 3:
        add_literal(tokens, labels, "நிலத்தின் வகை")
        add_entity(tokens, labels, land_type, "LAND_TYPE")
        add_punctuation(tokens, labels, ".")


def add_year(record, year, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    year = str(year)

    if style == 0:
        add_literal(tokens, labels, "ஆண்டு")
        add_entity(tokens, labels, year, "YEAR")
        add_punctuation(tokens, labels, ",")

    elif style == 1:
        add_literal(tokens, labels, "வருடம்")
        add_entity(tokens, labels, year, "YEAR")
        add_punctuation(tokens, labels, ",")

    elif style == 2:
        add_literal(tokens, labels, "பதிவு ஆண்டு")
        add_entity(tokens, labels, year, "YEAR")
        add_punctuation(tokens, labels, ",")


def add_document_number(record, document_number, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    if style == 0:
        add_literal(tokens, labels, "ஆவண எண்")
        add_entity(tokens, labels, document_number, "DOCUMENT_NUMBER")
        add_punctuation(tokens, labels, ",")

    elif style == 1:
        add_literal(tokens, labels, "ஆவணம் எண்")
        add_entity(tokens, labels, document_number, "DOCUMENT_NUMBER")
        add_punctuation(tokens, labels, ",")

    elif style == 2:
        add_literal(tokens, labels, "ஆவண எண்:")
        add_entity(tokens, labels, document_number, "DOCUMENT_NUMBER")
        add_punctuation(tokens, labels, ",")


def add_boundary(record, direction, place, style=0):
    tokens = record["tokens"]
    labels = record["labels"]

    if style == 0:
        add_literal(tokens, labels, direction)
        add_literal(tokens, labels, "எல்லை")
        add_entity(tokens, labels, place, "BOUNDARY")
        add_punctuation(tokens, labels, ",")

    elif style == 1:
        add_literal(tokens, labels, direction)
        add_literal(tokens, labels, "பக்கம்")
        add_entity(tokens, labels, place, "BOUNDARY")
        add_punctuation(tokens, labels, ",")

    elif style == 2:
        add_literal(tokens, labels, direction)
        add_punctuation(tokens, labels, ":")
        add_entity(tokens, labels, place, "BOUNDARY")
        add_punctuation(tokens, labels, ",")


def add_boundaries(record):
    directions = [
        ("வடக்கு", random.choice(BOUNDARY_PLACES)),
        ("தெற்கு", random.choice(BOUNDARY_PLACES)),
        ("கிழக்கு", random.choice(BOUNDARY_PLACES)),
        ("மேற்கு", random.choice(BOUNDARY_PLACES)),
    ]

    random.shuffle(directions)

    style = random.randint(0, 2)

    for direction, place in directions:
        add_boundary(
            record,
            direction,
            place,
            style=random.randint(0, 2)
        )

    # End boundary section cleanly.
    if record["labels"] and record["tokens"][-1] == ",":
        record["tokens"][-1] = "."
        record["labels"][-1] = "O"


# ======================================================================
# HARD NEGATIVE / NON-ENTITY NUMERIC TEXT
# ======================================================================

def add_hard_negative_numeric(record):
    """
    Add numbers that should NOT become land entities.

    This helps prevent the model from blindly classifying every number
    as PATTA, SURVEY, YEAR, or AREA.
    """

    tokens = record["tokens"]
    labels = record["labels"]

    negative_patterns = [
        f"மொத்தம் {random.randint(1, 999)} குடும்பங்கள்",
        f"கதவு எண் {random.randint(1, 999)}",
        f"தொலைபேசி {random.randint(6000000000, 9999999999)}",
        f"பக்கம் {random.randint(1, 30)}",
        f"வகுப்பு {random.randint(1, 12)}",
        f"கோப்பு {random.randint(1000, 9999)}",
        f"பதிவு பக்கம் {random.randint(1, 500)}",
        f"வரி ஆண்டு {random.randint(2010, 2026)}",
    ]

    add_literal(
        tokens,
        labels,
        random.choice(negative_patterns)
    )

    add_punctuation(tokens, labels, ",")


# ======================================================================
# RECORD TYPES
# ======================================================================

def make_short_record():
    """
    Short, simple records.
    """

    record = new_record()

    owner = random_name()
    father = random_name(exclude=owner)

    village = random_village()
    taluk = random_taluk()
    district = random_district()

    survey = random_survey_number()
    patta = random_patta_number()

    area = random_simple_area()
    land_type = random.choice(LAND_TYPES)

    style = random.randint(0, 3)

    add_owner(record, owner, father, style)
    add_survey(record, survey, random.randint(0, 3))
    add_patta(record, patta, random.randint(0, 3))
    add_location(record, village, taluk, district, random.randint(0, 3))
    add_area(record, area, random.randint(0, 3))
    add_land_type(record, land_type, random.randint(0, 3))

    return record


def make_medium_record():
    """
    Medium realistic land-record descriptions.
    """

    record = new_record()

    owner = random_name()
    father = random_name(exclude=owner)

    village = random_village()
    taluk = random_taluk()
    district = random_district()

    survey = random_survey_number()
    patta = random_patta_number()

    area = random_area()
    land_type = random.choice(LAND_TYPES)

    components = [
        ("owner", owner),
        ("father", father),
        ("survey", survey),
        ("patta", patta),
        ("location", (village, taluk, district)),
        ("area", area),
        ("land_type", land_type),
    ]

    random.shuffle(components)

    add_literal(
        record["tokens"],
        record["labels"],
        random.choice([
            "இந்நிலத்தின் விவரங்கள் பின்வருமாறு",
            "நிலப்பதிவில் குறிப்பிடப்பட்ட விவரங்கள்",
            "உரிமை மற்றும் நில விவரங்கள்",
            "பதிவில் காணப்படும் நில விவரங்கள்",
        ])
    )

    add_punctuation(record["tokens"], record["labels"], ":")

    for component, value in components:

        if component == "owner":
            add_owner(
                record,
                value,
                father,
                random.randint(0, 3)
            )

        elif component == "father":
            # Sometimes father appears independently.
            add_literal(
                record["tokens"],
                record["labels"],
                random.choice([
                    "தந்தை பெயர்",
                    "தந்தை",
                    "த/பெ",
                ])
            )

            add_entity(
                record["tokens"],
                record["labels"],
                value,
                "FATHER_NAME"
            )

            add_punctuation(
                record["tokens"],
                record["labels"],
                ","
            )

        elif component == "survey":
            add_survey(
                record,
                value,
                random.randint(0, 3)
            )

        elif component == "patta":
            add_patta(
                record,
                value,
                random.randint(0, 3)
            )

        elif component == "location":
            village_v, taluk_v, district_v = value

            add_location(
                record,
                village_v,
                taluk_v,
                district_v,
                random.randint(0, 3)
            )

        elif component == "area":
            add_area(
                record,
                value,
                random.randint(0, 3)
            )

        elif component == "land_type":
            add_land_type(
                record,
                value,
                random.randint(0, 3)
            )

    return record


def make_multi_parcel_record():
    """
    Multiple land parcels belonging to one or more owners.
    """

    record = new_record()

    owner_count = random.choice([1, 1, 2])

    owners = []

    for _ in range(owner_count):
        owner = random_name(
            exclude=owners[-1] if owners else None
        )

        father = random_name(exclude=owner)

        owners.append(
            (owner, father)
        )

    village = random_village()
    taluk = random_taluk()
    district = random_district()

    add_literal(
        record["tokens"],
        record["labels"],
        random.choice([
            "நில உரிமை விவரங்களின்படி",
            "குறிப்பிட்ட பதிவின்படி",
            "இப்பதிவில்",
            "கீழ்கண்ட நில விவரங்கள் பதிவு செய்யப்பட்டுள்ளன",
        ])
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        ":"
    )

    for index, (owner, father) in enumerate(owners):

        if index > 0:
            add_literal(
                record["tokens"],
                record["labels"],
                random.choice([
                    "மேலும் உரிமையாளர்",
                    "இரண்டாவது உரிமையாளர்",
                    "கூடுதல் உரிமையாளர்",
                ])
            )

            add_punctuation(
                record["tokens"],
                record["labels"],
                ":"
            )

        add_entity(
            record["tokens"],
            record["labels"],
            owner,
            "OWNER"
        )

        add_literal(
            record["tokens"],
            record["labels"],
            random.choice([
                "தந்தை",
                "தந்தை பெயர்",
                "த/பெ",
            ])
        )

        add_entity(
            record["tokens"],
            record["labels"],
            father,
            "FATHER_NAME"
        )

        add_punctuation(
            record["tokens"],
            record["labels"],
            ","
        )

    add_location(
        record,
        village,
        taluk,
        district,
        random.randint(0, 3)
    )

    parcel_count = random.randint(2, 5)

    for i in range(parcel_count):

        survey = random_survey_number()
        patta = random_patta_number()
        area = random_area()
        land_type = random.choice(LAND_TYPES)

        add_literal(
            record["tokens"],
            record["labels"],
            random.choice([
                "நிலப்பகுதி",
                "மனைப்பகுதி",
                "பகுதி",
                "குறிப்பிட்ட நிலம்",
            ])
        )

        add_literal(
            record["tokens"],
            record["labels"],
            str(i + 1)
        )

        add_punctuation(
            record["tokens"],
            record["labels"],
            ":"
        )

        add_survey(
            record,
            survey,
            random.randint(0, 3)
        )

        add_patta(
            record,
            patta,
            random.randint(0, 3)
        )

        add_area(
            record,
            area,
            random.randint(0, 3)
        )

        add_land_type(
            record,
            land_type,
            random.randint(0, 3)
        )

        if random.random() < 0.4:
            add_literal(
                record["tokens"],
                record["labels"],
                random.choice([
                    "இப்பகுதிக்கு தனித்த பதிவு உள்ளது",
                    "இந்நிலப்பகுதி தனியாக குறிப்பிடப்பட்டுள்ளது",
                    "இப்பகுதி உரிமை பதிவில் உள்ளது",
                ])
            )

            add_punctuation(
                record["tokens"],
                record["labels"],
                "."
            )

    return record


def make_long_record():
    """
    Long realistic paragraph designed to stress-test the NER model.
    """

    record = new_record()

    owner = random_name()
    father = random_name(exclude=owner)

    second_owner = random_name(
        exclude=owner
    )

    second_father = random_name(
        exclude=second_owner
    )

    village = random_village()
    taluk = random_taluk()
    district = random_district()

    add_literal(
        record["tokens"],
        record["labels"],
        random.choice([
            "இப்பதிவின்படி கீழ்க்கண்ட நிலம்",
            "பதிவில் காணப்படும் நில உரிமை மற்றும் சொத்து விவரங்களின்படி",
            "குறிப்பிட்ட ஆவணத்தின் அடிப்படையில் பதிவு செய்யப்பட்ட நில விவரம்",
            "வருவாய் பதிவில் உள்ள விவரங்களின்படி",
        ])
    )

    add_literal(
        record["tokens"],
        record["labels"],
        random.choice([
            "உரிமையாளர்",
            "நில உரிமையாளர்",
            "சொத்து உரிமையாளர்",
        ])
    )

    add_entity(
        record["tokens"],
        record["labels"],
        owner,
        "OWNER"
    )

    add_literal(
        record["tokens"],
        record["labels"],
        "தந்தை பெயர்"
    )

    add_entity(
        record["tokens"],
        record["labels"],
        father,
        "FATHER_NAME"
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        ","
    )

    # Main location.
    add_location(
        record,
        village,
        taluk,
        district,
        random.randint(0, 3)
    )

    # First parcel.
    survey1 = random_survey_number()
    patta1 = random_patta_number()
    area1 = random_area()
    land_type1 = random.choice(LAND_TYPES)

    add_literal(
        record["tokens"],
        record["labels"],
        "இந்த நிலத்திற்கு"
    )

    add_survey(
        record,
        survey1,
        random.randint(0, 3)
    )

    add_patta(
        record,
        patta1,
        random.randint(0, 3)
    )

    add_area(
        record,
        area1,
        random.randint(0, 3)
    )

    add_land_type(
        record,
        land_type1,
        random.randint(0, 3)
    )

    # Second parcel.
    survey2 = random_survey_number()
    patta2 = random_patta_number()
    area2 = random_area()
    land_type2 = random.choice(LAND_TYPES)

    add_literal(
        record["tokens"],
        record["labels"],
        random.choice([
            "மேலும் இதனுடன் தொடர்புடைய",
            "அதே உரிமையாளருக்குச் சேர்ந்த மற்றொரு",
            "கூடுதலாக",
            "இதனுடன் இணைந்த",
        ])
    )

    add_survey(
        record,
        survey2,
        random.randint(0, 3)
    )

    add_patta(
        record,
        patta2,
        random.randint(0, 3)
    )

    add_area(
        record,
        area2,
        random.randint(0, 3)
    )

    add_land_type(
        record,
        land_type2,
        random.randint(0, 3)
    )

    # Optional second owner.
    if random.random() < 0.65:

        add_literal(
            record["tokens"],
            record["labels"],
            random.choice([
                "கூட்டு உரிமையாளராக",
                "மற்றொரு உரிமையாளராக",
                "கூடுதல் உரிமையாளராக",
            ])
        )

        add_entity(
            record["tokens"],
            record["labels"],
            second_owner,
            "OWNER"
        )

        add_literal(
            record["tokens"],
            record["labels"],
            "தந்தை பெயர்"
        )

        add_entity(
            record["tokens"],
            record["labels"],
            second_father,
            "FATHER_NAME"
        )

        add_punctuation(
            record["tokens"],
            record["labels"],
            ","
        )

    # Boundary section.
    add_literal(
        record["tokens"],
        record["labels"],
        random.choice([
            "நிலத்தின் எல்லைகள் பின்வருமாறு",
            "சொத்து நான்கு புற எல்லைகளால் வரையறுக்கப்பட்டுள்ளது",
            "நில எல்லை விவரங்கள்",
            "நிலத்தின் நான்கு திசை எல்லைகள்",
        ])
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        ":"
    )

    add_boundaries(record)

    # Document metadata.
    document_number = random_document_number()
    year = random_year()

    add_document_number(
        record,
        document_number,
        random.randint(0, 2)
    )

    add_year(
        record,
        year,
        random.randint(0, 2)
    )

    # Hard negatives.
    if random.random() < 0.45:
        add_hard_negative_numeric(record)

    # Final statement.
    add_literal(
        record["tokens"],
        record["labels"],
        random.choice([
            "மேற்கண்ட விவரங்கள் பதிவில் காணப்படுகின்றன",
            "மேற்கண்ட நில விவரங்கள் ஆவணத்தில் பதிவு செய்யப்பட்டுள்ளன",
            "இந்த விவரங்கள் தொடர்புடைய பதிவில் குறிப்பிடப்பட்டுள்ளன",
            "சொத்து விவரங்கள் பதிவேட்டில் காணப்படுகின்றன",
        ])
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        "."
    )

    return record


def make_metadata_heavy_record():
    """
    Record emphasizing document/year/patta/survey ambiguity.
    """

    record = new_record()

    owner = random_name()
    father = random_name(exclude=owner)

    village = random_village()
    taluk = random_taluk()
    district = random_district()

    document_number = random_document_number()
    year = random_year()

    survey1 = random_survey_number()
    survey2 = random_survey_number()

    patta1 = random_patta_number()
    patta2 = random_patta_number()

    add_literal(
        record["tokens"],
        record["labels"],
        "ஆவண விவரங்கள்"
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        ":"
    )

    add_document_number(
        record,
        document_number,
        random.randint(0, 2)
    )

    add_year(
        record,
        year,
        random.randint(0, 2)
    )

    add_literal(
        record["tokens"],
        record["labels"],
        "உரிமையாளர்"
    )

    add_entity(
        record["tokens"],
        record["labels"],
        owner,
        "OWNER"
    )

    add_literal(
        record["tokens"],
        record["labels"],
        "தந்தை"
    )

    add_entity(
        record["tokens"],
        record["labels"],
        father,
        "FATHER_NAME"
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        ","
    )

    add_location(
        record,
        village,
        taluk,
        district,
        random.randint(0, 3)
    )

    add_literal(
        record["tokens"],
        record["labels"],
        "முதன்மை சர்வே"
    )

    add_entity(
        record["tokens"],
        record["labels"],
        survey1,
        "SURVEY_NUMBER"
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        ","
    )

    add_literal(
        record["tokens"],
        record["labels"],
        "துணை சர்வே"
    )

    add_entity(
        record["tokens"],
        record["labels"],
        survey2,
        "SURVEY_NUMBER"
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        ","
    )

    add_literal(
        record["tokens"],
        record["labels"],
        "பட்டா விவரம்"
    )

    add_entity(
        record["tokens"],
        record["labels"],
        patta1,
        "PATTA_NUMBER"
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        ","
    )

    add_entity(
        record["tokens"],
        record["labels"],
        patta2,
        "PATTA_NUMBER"
    )

    add_punctuation(
        record["tokens"],
        record["labels"],
        "."
    )

    return record


# ======================================================================
# DATASET MIXING
# ======================================================================

def generate_record():
    """
    Weighted V3 mixture.

    Short       20%
    Medium      30%
    Multi       15%
    Long        25%
    Metadata    10%
    """

    r = random.random()

    if r < 0.20:
        return make_short_record()

    if r < 0.50:
        return make_medium_record()

    if r < 0.65:
        return make_multi_parcel_record()

    if r < 0.90:
        return make_long_record()

    return make_metadata_heavy_record()


# ======================================================================
# VALIDATION
# ======================================================================

def validate_record(record):
    if not isinstance(record, dict):
        return False

    if "tokens" not in record:
        return False

    if "labels" not in record:
        return False

    tokens = record["tokens"]
    labels = record["labels"]

    if not isinstance(tokens, list):
        return False

    if not isinstance(labels, list):
        return False

    if len(tokens) != len(labels):
        return False

    if len(tokens) == 0:
        return False

    for token in tokens:
        if not isinstance(token, str):
            return False

        if not token.strip():
            return False

    for label in labels:
        if label not in LABEL_TO_ID:
            return False

    # Validate BIO consistency.
    previous_entity = None

    for label in labels:

        if label == "O":
            previous_entity = None
            continue

        if label.startswith("B-"):
            previous_entity = label[2:]
            continue

        if label.startswith("I-"):
            entity_type = label[2:]

            if previous_entity != entity_type:
                return False

            previous_entity = entity_type

    return True


def validate_dataset(dataset, name):
    if not isinstance(dataset, list):
        raise RuntimeError(
            f"{name} dataset must be a list."
        )

    if len(dataset) == 0:
        raise RuntimeError(
            f"{name} dataset is empty."
        )

    for index, record in enumerate(dataset):

        if not validate_record(record):
            raise RuntimeError(
                f"{name} dataset validation failed at record {index}."
            )

    print(
        f"  {name:<12} OK ({len(dataset):,} examples)"
    )


# ======================================================================
# DUPLICATE CONTROL
# ======================================================================

def record_signature(record):
    return (
        " ".join(record["tokens"])
        + "|||"
        + " ".join(record["labels"])
    )


def generate_unique_dataset(size, dataset_name):
    dataset = []
    signatures = set()

    attempts = 0
    max_attempts = size * 10

    while len(dataset) < size:

        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate enough unique examples "
                f"for {dataset_name}."
            )

        record = generate_record()

        if not validate_record(record):
            continue

        signature = record_signature(record)

        if signature in signatures:
            continue

        signatures.add(signature)
        dataset.append(record)

        if len(dataset) % 5000 == 0 or len(dataset) == size:
            print(
                f"  {dataset_name:<12}: "
                f"{len(dataset):,}/{size:,}"
            )

    return dataset


# ======================================================================
# SAVE
# ======================================================================

def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def save_labels():
    save_json(
        OUTPUT_DIR / "labels.json",
        {
            "labels": LABELS,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": {
                str(k): v
                for k, v in ID_TO_LABEL.items()
            },
            "num_labels": len(LABELS),
        }
    )


# ======================================================================
# STATISTICS
# ======================================================================

def entity_statistics(dataset):
    counts = {
        label: 0
        for label in LABELS
        if label.startswith("B-")
    }

    for record in dataset:

        for label in record["labels"]:

            if label.startswith("B-"):
                counts[label] += 1

    return counts


def print_statistics(dataset, name):
    counts = entity_statistics(dataset)

    print()
    print(f"{name} entity statistics")
    print("-" * 70)

    for label in counts:
        print(
            f"{label:<25} {counts[label]:>8,}"
        )

    total_entities = sum(counts.values())

    print("-" * 70)
    print(
        f"{'TOTAL ENTITIES':<25} {total_entities:>8,}"
    )


# ======================================================================
# EXAMPLES
# ======================================================================

def print_examples(dataset):
    print()
    print("=" * 70)
    print("V3 DATASET EXAMPLES")
    print("=" * 70)

    sample_indices = random.sample(
        range(len(dataset)),
        min(8, len(dataset))
    )

    for index in sample_indices:

        record = dataset[index]

        print()
        print("TEXT:")
        print(
            " ".join(record["tokens"])
        )

        print()
        print("LABELS:")

        for token, label in zip(
            record["tokens"],
            record["labels"]
        ):

            if label != "O":
                print(
                    f"  {token:<25} {label}"
                )

        print("-" * 70)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("       TAMIL LAND RECORD NER - DATASET V3")
    print("=" * 70)

    print()
    print("Configuration:")
    print(
        f"  Training:    {TRAIN_SIZE:,}"
    )
    print(
        f"  Validation:  {VALIDATION_SIZE:,}"
    )
    print(
        f"  Test:        {TEST_SIZE:,}"
    )
    print(
        f"  Total:       "
        f"{TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE:,}"
    )

    print()
    print("Output directory:")
    print(
        f"  {OUTPUT_DIR.resolve()}"
    )

    print()
    print("Generating V3 datasets...")
    print()

    train = generate_unique_dataset(
        TRAIN_SIZE,
        "Training"
    )

    validation = generate_unique_dataset(
        VALIDATION_SIZE,
        "Validation"
    )

    test = generate_unique_dataset(
        TEST_SIZE,
        "Test"
    )

    print()
    print("Validating datasets...")

    validate_dataset(
        train,
        "Training"
    )

    validate_dataset(
        validation,
        "Validation"
    )

    validate_dataset(
        test,
        "Test"
    )

    # Make sure there is no overlap.
    train_signatures = {
        record_signature(x)
        for x in train
    }

    validation_signatures = {
        record_signature(x)
        for x in validation
    }

    test_signatures = {
        record_signature(x)
        for x in test
    }

    if train_signatures & validation_signatures:
        raise RuntimeError(
            "Train/validation duplicate detected."
        )

    if train_signatures & test_signatures:
        raise RuntimeError(
            "Train/test duplicate detected."
        )

    if validation_signatures & test_signatures:
        raise RuntimeError(
            "Validation/test duplicate detected."
        )

    print()
    print("Dataset overlap check: OK")

    # Save datasets.
    print()
    print("Saving datasets...")

    save_json(
        OUTPUT_DIR / "train.json",
        train
    )

    save_json(
        OUTPUT_DIR / "validation.json",
        validation
    )

    save_json(
        OUTPUT_DIR / "test.json",
        test
    )

    save_labels()

    print()
    print("Dataset files saved:")
    print(
        f"  {OUTPUT_DIR / 'train.json'}"
    )
    print(
        f"  {OUTPUT_DIR / 'validation.json'}"
    )
    print(
        f"  {OUTPUT_DIR / 'test.json'}"
    )
    print(
        f"  {OUTPUT_DIR / 'labels.json'}"
    )

    # Statistics.
    print_statistics(
        train,
        "TRAINING"
    )

    print_statistics(
        validation,
        "VALIDATION"
    )

    print_statistics(
        test,
        "TEST"
    )

    # Examples.
    print_examples(train)

    # Final summary.
    print()
    print("=" * 70)
    print("       DATASET V3 GENERATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Dataset sizes:")
    print(
        f"  Training:    {len(train):,}"
    )
    print(
        f"  Validation:  {len(validation):,}"
    )
    print(
        f"  Test:        {len(test):,}"
    )

    print()
    print("Labels:")
    for i, label in enumerate(LABELS):
        print(
            f"  {i:2d} -> {label}"
        )

    print()
    print("Next step:")
    print()
    print(
        "  Inspect the generated examples before training."
    )
    print()
    print(
        "  After validation, upload data/ner_v3 to Google Drive"
    )
    print(
        "  and train IndicBERT V3 in Google Colab."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()