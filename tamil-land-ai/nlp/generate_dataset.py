import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path("data/ner")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Tamil land-record vocabulary
# ---------------------------------------------------------

NAMES = [
    "ராமசாமி",
    "சுப்பிரமணியன்",
    "முருகன்",
    "கண்ணன்",
    "சின்னசாமி",
    "பெருமாள்",
    "கிருஷ்ணன்",
    "வேலாயுதம்",
    "முத்துசாமி",
    "செல்வம்",
    "துரைசாமி",
    "நடராஜன்",
    "சண்முகம்",
    "அருணாசலம்",
    "பழனிசாமி",
]

VILLAGES = [
    "மணப்பாறை",
    "லால்குடி",
    "திருவெறும்பூர்",
    "ஸ்ரீரங்கம்",
    "முசிறி",
    "துறையூர்",
    "குளித்தலை",
    "கரூர்",
    "பெரம்பலூர்",
    "அரியலூர்",
]

TALUKS = [
    "லால்குடி",
    "மணப்பாறை",
    "ஸ்ரீரங்கம்",
    "முசிறி",
    "துறையூர்",
    "குளித்தலை",
    "கரூர்",
]

DISTRICTS = [
    "திருச்சிராப்பள்ளி",
    "கரூர்",
    "பெரம்பலூர்",
    "அரியலூர்",
    "தஞ்சாவூர்",
    "புதுக்கோட்டை",
]

LAND_TYPES = [
    "நஞ்சை",
    "புஞ்சை",
    "தரிசு",
    "நன்செய்",
    "புன்செய்",
]

AREAS = [
    "1 ஏக்கர்",
    "2 ஏக்கர்",
    "3 ஏக்கர்",
    "1 ஏக்கர் 20 சென்ட்",
    "2 ஏக்கர் 15 சென்ட்",
    "2 ஏக்கர் 17 சென்ட்",
    "3 ஏக்கர் 25 சென்ட்",
    "50 சென்ட்",
    "75 சென்ட்",
]

DIRECTIONS = [
    "வடக்கு",
    "தெற்கு",
    "கிழக்கு",
    "மேற்கு",
]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def add_entity(tokens, labels, text, entity_type):
    """
    Add a multi-word entity using BIO tagging.
    """

    words = text.split()

    for i, word in enumerate(words):
        tokens.append(word)

        if i == 0:
            labels.append(f"B-{entity_type}")
        else:
            labels.append(f"I-{entity_type}")


def add_plain(tokens, labels, text):
    """
    Add normal/non-entity words.
    """

    words = text.split()

    for word in words:
        tokens.append(word)
        labels.append("O")


def survey_number():
    return f"{random.randint(1, 999)}/{random.randint(1, 9)}"


def patta_number():
    return str(random.randint(100, 99999))


def year():
    return str(random.randint(1900, 2025))


# ---------------------------------------------------------
# Sentence generators
# ---------------------------------------------------------

def template_owner():
    owner = random.choice(NAMES)
    father = random.choice([x for x in NAMES if x != owner])
    survey = survey_number()
    area = random.choice(AREAS)
    land_type = random.choice(LAND_TYPES)

    tokens = []
    labels = []

    add_entity(tokens, labels, owner, "OWNER")

    add_plain(tokens, labels, "மகன்")

    add_entity(tokens, labels, father, "FATHER_NAME")

    add_plain(tokens, labels, "என்பவருக்கு")

    add_plain(tokens, labels, "சர்வே")

    add_plain(tokens, labels, "எண்")

    add_entity(tokens, labels, survey, "SURVEY_NUMBER")

    add_plain(tokens, labels, "ல்")

    add_entity(tokens, labels, area, "AREA")

    add_entity(tokens, labels, land_type, "LAND_TYPE")

    add_plain(tokens, labels, "நிலம் உள்ளது")

    return tokens, labels


def template_patta():
    owner = random.choice(NAMES)
    patta = patta_number()
    survey = survey_number()
    village = random.choice(VILLAGES)

    tokens = []
    labels = []

    add_plain(tokens, labels, "பட்டா")

    add_plain(tokens, labels, "எண்")

    add_entity(tokens, labels, patta, "PATTA_NUMBER")

    add_plain(tokens, labels, "உரிமையாளர்")

    add_entity(tokens, labels, owner, "OWNER")

    add_plain(tokens, labels, "சர்வே")

    add_plain(tokens, labels, "எண்")

    add_entity(tokens, labels, survey, "SURVEY_NUMBER")

    add_plain(tokens, labels, "கிராமம்")

    add_entity(tokens, labels, village, "VILLAGE")

    return tokens, labels


def template_location():
    village = random.choice(VILLAGES)
    taluk = random.choice(TALUKS)
    district = random.choice(DISTRICTS)

    tokens = []
    labels = []

    add_plain(tokens, labels, "கிராமம்")

    add_entity(tokens, labels, village, "VILLAGE")

    add_plain(tokens, labels, "தாலுகா")

    add_entity(tokens, labels, taluk, "TALUK")

    add_plain(tokens, labels, "மாவட்டம்")

    add_entity(tokens, labels, district, "DISTRICT")

    return tokens, labels


def template_document():
    document_number = str(random.randint(100, 9999))
    date_year = year()

    tokens = []
    labels = []

    add_plain(tokens, labels, "ஆவண")

    add_plain(tokens, labels, "எண்")

    add_entity(tokens, labels, document_number, "DOCUMENT_NUMBER")

    add_plain(tokens, labels, "ஆண்டு")

    add_entity(tokens, labels, date_year, "YEAR")

    return tokens, labels


def template_boundary():
    owner = random.choice(NAMES)
    north = random.choice(NAMES)
    south = random.choice(NAMES)

    tokens = []
    labels = []

    add_entity(tokens, labels, owner, "OWNER")

    add_plain(tokens, labels, "நிலத்தின்")

    add_plain(tokens, labels, "எல்லைகள்")

    add_plain(tokens, labels, "வடக்கு")

    add_entity(tokens, labels, north, "BOUNDARY")

    add_plain(tokens, labels, "தெற்கு")

    add_entity(tokens, labels, south, "BOUNDARY")

    return tokens, labels


TEMPLATES = [
    template_owner,
    template_patta,
    template_location,
    template_document,
    template_boundary,
]


# ---------------------------------------------------------
# Generate dataset
# ---------------------------------------------------------

def generate_examples(count):

    examples = []

    for _ in range(count):

        generator = random.choice(TEMPLATES)

        tokens, labels = generator()

        examples.append({
            "tokens": tokens,
            "ner_tags": labels
        })

    return examples


# ---------------------------------------------------------
# Split
# ---------------------------------------------------------

TOTAL = 10000

examples = generate_examples(TOTAL)

random.shuffle(examples)

train_size = int(len(examples) * 0.8)
validation_size = int(len(examples) * 0.1)

train = examples[:train_size]

validation = examples[
    train_size:
    train_size + validation_size
]

test = examples[
    train_size + validation_size:
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "validation.json", "w", encoding="utf-8") as f:
    json.dump(validation, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "test.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)


print("Dataset created successfully!")

print(f"Total examples: {TOTAL}")
print(f"Training:       {len(train)}")
print(f"Validation:     {len(validation)}")
print(f"Test:           {len(test)}")