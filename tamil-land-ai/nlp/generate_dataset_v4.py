"""
V4 Tamil Land Record NER dataset generator.

Training-only design:
- Focuses on the original/core land-record fields instead of 50+ labels.
- Strongly separates PATTA_NUMBER / SURVEY_NUMBER / DOCUMENT_NUMBER.
- Adds DATE, LATITUDE and LONGITUDE explicitly.
- Uses Tamil, English and mixed Tamil-English templates.
- Generates hard-negative numbers/IDs that must remain O.
- Uses exact character spans before tokenization.
- Produces reproducible train/validation/test splits.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

SEED = 20260909
TRAIN_SIZE = 30000
VALIDATION_SIZE = 4000
TEST_SIZE = 4000
OUT = Path("data") / "ner_v4"

ENTITY_TYPES = [
    "OWNER",
    "FATHER_NAME",
    "PATTA_NUMBER",
    "SURVEY_NUMBER",
    "SUBDIVISION_NUMBER",
    "DOCUMENT_NUMBER",
    "VILLAGE",
    "TALUK",
    "DISTRICT",
    "AREA",
    "LAND_TYPE",
    "BOUNDARY",
    "DATE",
    "YEAR",
    "LATITUDE",
    "LONGITUDE",
]

LABELS = ["O"] + [
    f"{prefix}-{entity}"
    for entity in ENTITY_TYPES
    for prefix in ("B", "I")
]

rng = random.Random(SEED)

NAMES_TA = [
    "ராமசாமி", "முருகேசன்", "செந்தில்குமார்", "சுப்பிரமணியன்",
    "பழனிச்சாமி", "கண்ணன்", "கோபால்", "சிவக்குமார்", "மணிகண்டன்",
    "ராஜேந்திரன்", "வெங்கடேசன்", "சரவணன்", "மோகன்", "குமார்",
    "சுரேஷ்", "மகேந்திரன்", "அருணாசலம்", "துரைசாமி", "நடராஜன்",
    "ஜெயக்குமார்", "பிரபாகரன்", "விஜயகுமார்", "தனபால்", "சண்முகம்",
    "கிருஷ்ணமூர்த்தி", "ரமேஷ்", "அன்பழகன்", "சக்திவேல்", "கார்த்திகேயன்",
    "லட்சுமி", "மீனாட்சி", "கலையரசி", "பிரியா", "மாலதி",
]

NAMES_EN = [
    "Suresh Kumar", "Rajendran", "Murugan", "Karthikeyan",
    "Ramesh Kumar", "Lakshmi Devi", "Shanmugam", "Senthil Kumar",
    "Priya", "Arun Kumar", "Vijay Kumar", "Meena",
]

DISTRICTS_TA = [
    "கோயம்புத்தூர்", "திருச்சிராப்பள்ளி", "மதுரை", "திண்டுக்கல்",
    "திருப்பூர்", "ஈரோடு", "சேலம்", "கரூர்", "நாமக்கல்",
    "திருநெல்வேலி", "தூத்துக்குடி", "விருதுநகர்", "தேனி",
    "தஞ்சாவூர்", "கடலூர்", "செங்கல்பட்டு", "திருவள்ளூர்", "வேலூர்",
]

DISTRICTS_EN = [
    "Coimbatore", "Tiruchirappalli", "Madurai", "Dindigul",
    "Tiruppur", "Erode", "Salem", "Karur", "Namakkal",
    "Tirunelveli", "Thoothukudi", "Virudhunagar", "Theni",
    "Thanjavur", "Cuddalore", "Chengalpattu", "Tiruvallur", "Vellore",
]

TALUKS_TA = [
    "பொள்ளாச்சி", "கோவை தெற்கு", "கோவை வடக்கு", "மேட்டுப்பாளையம்",
    "சூலூர்", "உடுமலைப்பேட்டை", "வால்பாறை", "திருப்பூர்", "பல்லடம்",
    "அவிநாசி", "பழனி", "திண்டுக்கல்", "கரூர்", "நாமக்கல்", "சேலம்",
    "ஈரோடு", "கோபிசெட்டிபாளையம்", "திருச்செங்கோடு",
]

TALUKS_EN = [
    "Pollachi", "Coimbatore South", "Coimbatore North", "Mettupalayam",
    "Sulur", "Udumalpet", "Valparai", "Tiruppur", "Palladam",
    "Avinashi", "Palani", "Dindigul", "Karur", "Namakkal", "Salem", "Erode",
]

VILLAGES_TA = [
    "ஆனைமலை", "காளப்பட்டி", "சோமந்துறை", "புத்தாநத்தம்", "செட்டிபாளையம்",
    "கிணத்துக்கடவு", "மதுக்கரை", "சுல்தான்பேட்டை", "கோவில்பாளையம்",
    "வடக்கலூர்", "வெள்ளலூர்", "சரவணம்பட்டி", "குனியமுத்தூர்", "பேரூர்",
    "நாச்சிபாளையம்", "வால்பாறை", "உடுமலை", "குரும்பபாளையம்", "துடியலூர்",
    "அன்னூர்", "மேட்டுப்பாளையம்", "சிங்காநல்லூர்", "சூலூர்", "காரமடை",
    "பெரியநாயக்கன்பாளையம்", "இடையர்பாளையம்",
]

VILLAGES_EN = [
    "Kinathukadavu", "Kalapatti", "Anaimalai", "Pollachi", "Madukkarai",
    "Sulur", "Annur", "Perur", "Kovilpalayam", "Saravanampatti",
    "Vellalore", "Thudiyalur", "Mettupalayam", "Karamadai",
]

LAND_TA = ["நஞ்சை", "புஞ்சை", "மானாவாரி நிலம்", "நத்தம்", "தோட்ட நிலம்"]
LAND_EN = ["wet land", "dry land", "agricultural land", "residential land", "garden land"]

BOUNDARY_TA = [
    "முக்கிய சாலை", "கால்வாய்", "வாய்க்கால்", "அரசு புறம்போக்கு நிலம்",
    "அண்டை நிலம்", "ஏரி", "ஆறு",
]
BOUNDARY_EN = [
    "Main Road", "Canal", "Irrigation Channel", "Government Poramboke Land",
    "Adjacent Land", "Lake", "River",
]

LATITUDES = ["10.9825", "11.0168", "11.3410", "10.7905", "10.9601", "11.1085"]
LONGITUDES = ["76.9558", "76.995164", "78.1198", "77.3411", "78.7047", "77.5887"]


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def pick(values):
    return norm(rng.choice(values))


def date_value():
    return f"{rng.randint(1,28):02d}-{rng.randint(1,12):02d}-{rng.randint(1990,2026)}"


def year_value():
    return str(rng.randint(1950, 2026))


def survey_value():
    base = rng.randint(1, 9999)
    sub = rng.randint(1, 99)
    suffix = rng.choice(["", "A", "B", "C", "A1", "B1", "A2", "B2"])
    return f"{base}/{sub}{suffix}"


def subdivision_value():
    return rng.choice([
        str(rng.randint(1, 99)),
        f"{rng.randint(1, 99)}A",
        f"{rng.randint(1, 99)}B",
        f"{rng.randint(1, 99)}A1",
        f"{rng.randint(1, 99)}B1",
    ])


def patta_value():
    return str(rng.randint(1000, 999999))


def document_value():
    return rng.choice([
        f"{rng.randint(1,9999)}/{rng.randint(1990,2026)}",
        f"{rng.randint(10000,999999)}",
        f"DOC/{rng.choice(['CBR','MDU','TPR','SLM'])}/{rng.randint(2010,2026)}/{rng.randint(1,99999):05d}",
        f"DOC-{rng.randint(1,999)}/{rng.randint(2010,2026)}",
    ])


def area_value():
    return rng.choice([
        f"{rng.randint(1,9)}.{rng.randint(1,99):02d} acres",
        f"{rng.randint(1,9)} acres",
        f"{rng.randint(10,99)} cents",
        f"{rng.randint(1,9)} acres {rng.randint(1,99)} cents",
        f"{rng.randint(500,20000)} sq ft",
        f"{rng.randint(1,8)}.{rng.randint(1,99):02d} ஏக்கர்",
        f"{rng.randint(10,99)} சென்ட்",
    ])


class Record:
    def __init__(self):
        self.text = ""
        self.spans = []

    def add(self, value: str, entity: str | None = None):
        value = norm(value)
        if self.text and not self.text.endswith((" ", "\n")):
            self.text += " "
        start = len(self.text)
        self.text += value
        if entity:
            self.spans.append((start, len(self.text), entity))

    def line(self, *parts):
        if self.text:
            self.text += "\n"
        for value, entity in parts:
            self.add(value, entity)


def make_record():
    r = Record()

    owner = pick(NAMES_TA + NAMES_EN)
    father = pick(NAMES_TA + NAMES_EN)
    district = pick(DISTRICTS_TA + DISTRICTS_EN)
    taluk = pick(TALUKS_TA + TALUKS_EN)
    village = pick(VILLAGES_TA + VILLAGES_EN)

    patta = patta_value()
    survey = survey_value()
    subdivision = subdivision_value()
    document = document_value()
    dt = date_value()
    yr = year_value()
    area = area_value()
    land = pick(LAND_TA + LAND_EN)
    boundary_n = pick(BOUNDARY_TA + BOUNDARY_EN)
    boundary_s = pick(BOUNDARY_TA + BOUNDARY_EN)
    lat = pick(LATITUDES)
    lon = pick(LONGITUDES)

    style = rng.choices(["ta", "en", "mix"], weights=[0.45, 0.20, 0.35])[0]

    if style == "ta":
        r.line(
            ("உரிமையாளர்", None), (owner, "OWNER"),
            ("தந்தை பெயர்", None), (father, "FATHER_NAME"),
        )
        r.line(
            ("மாவட்டம்", None), (district, "DISTRICT"),
            ("வட்டம்", None), (taluk, "TALUK"),
            ("கிராமம்", None), (village, "VILLAGE"),
        )
        r.line(
            ("பட்டா எண்", None), (patta, "PATTA_NUMBER"),
            ("சர்வே எண்", None), (survey, "SURVEY_NUMBER"),
            ("உட்பிரிவு எண்", None), (subdivision, "SUBDIVISION_NUMBER"),
        )
        r.line(
            ("ஆவண எண்", None), (document, "DOCUMENT_NUMBER"),
            ("தேதி", None), (dt, "DATE"),
            ("ஆண்டு", None), (yr, "YEAR"),
        )
        r.line(
            ("பரப்பளவு", None), (area, "AREA"),
            ("நில வகை", None), (land, "LAND_TYPE"),
        )
        r.line(
            ("வடக்கு", None), (boundary_n, "BOUNDARY"),
            ("தெற்கு", None), (boundary_s, "BOUNDARY"),
        )
        r.line(
            ("அட்சரேகை", None), (lat, "LATITUDE"),
            ("தீர்க்கரேகை", None), (lon, "LONGITUDE"),
        )

    elif style == "en":
        r.line(
            ("Owner", None), (owner, "OWNER"),
            ("Father Name", None), (father, "FATHER_NAME"),
        )
        r.line(
            ("District", None), (district, "DISTRICT"),
            ("Taluk", None), (taluk, "TALUK"),
            ("Village", None), (village, "VILLAGE"),
        )
        r.line(
            ("Patta Number", None), (patta, "PATTA_NUMBER"),
            ("Survey No.", None), (survey, "SURVEY_NUMBER"),
            ("Subdivision No.", None), (subdivision, "SUBDIVISION_NUMBER"),
        )
        r.line(
            ("Document No.", None), (document, "DOCUMENT_NUMBER"),
            ("Date", None), (dt, "DATE"),
            ("Year", None), (yr, "YEAR"),
        )
        r.line(
            ("Extent", None), (area, "AREA"),
            ("Land Type", None), (land, "LAND_TYPE"),
        )
        r.line(
            ("North", None), (boundary_n, "BOUNDARY"),
            ("South", None), (boundary_s, "BOUNDARY"),
        )
        r.line(
            ("Latitude", None), (lat, "LATITUDE"),
            ("Longitude", None), (lon, "LONGITUDE"),
        )

    else:
        # Mixed-language labels deliberately vary the context around the entities.
        r.line(
            ("உரிமையாளர் / Owner", None), (owner, "OWNER"),
            ("Father Name / தந்தை பெயர்", None), (father, "FATHER_NAME"),
        )
        r.line(
            ("District / மாவட்டம்", None), (district, "DISTRICT"),
            ("Taluk / வட்டம்", None), (taluk, "TALUK"),
            ("Village / கிராமம்", None), (village, "VILLAGE"),
        )
        r.line(
            ("Patta No.", None), (patta, "PATTA_NUMBER"),
            ("சர்வே எண் / Survey No.", None), (survey, "SURVEY_NUMBER"),
            ("Sub Division", None), (subdivision, "SUBDIVISION_NUMBER"),
        )
        r.line(
            ("Document No. / ஆவண எண்", None), (document, "DOCUMENT_NUMBER"),
            ("Date / தேதி", None), (dt, "DATE"),
            ("Year / ஆண்டு", None), (yr, "YEAR"),
        )
        r.line(
            ("Extent / பரப்பளவு", None), (area, "AREA"),
            ("Land Type / நில வகை", None), (land, "LAND_TYPE"),
        )
        r.line(
            ("North / வடக்கு", None), (boundary_n, "BOUNDARY"),
            ("South / தெற்கு", None), (boundary_s, "BOUNDARY"),
        )
        r.line(
            ("Latitude / அட்சரேகை", None), (lat, "LATITUDE"),
            ("Longitude / தீர்க்கரேகை", None), (lon, "LONGITUDE"),
        )

    # Hard negatives: numeric-looking strings which are intentionally O.
    # These are critical because V3 was over-predicting numbers as survey/patta values.
    r.line(
        ("Page No.", None), (str(rng.randint(1, 250)), None),
        ("File Ref.", None), (f"FILE-{rng.randint(1000,9999)}", None),
    )
    r.line(
        ("Receipt No.", None), (str(rng.randint(10000,999999)), None),
        ("Amount", None), (f"Rs.{rng.randint(100,99999)}", None),
    )
    r.line(
        ("Reference", None), (f"REF/{rng.randint(1,9999)}/{rng.randint(2010,2026)}", None),
        ("Code", None), (f"AB-{rng.randint(1000,9999)}", None),
    )

    # OCR-like context noise is applied to labels, never to gold entity values.
    # This teaches the model to rely on context plus entity form.
    if rng.random() < 0.35:
        replacements = {
            "Survey No.": rng.choice(["Survay No.", "Survey N0.", "Survey No"]),
            "Patta Number": rng.choice(["Patta Numbr", "Patta No", "Ptta Number"]),
            "Document No.": rng.choice(["Doc No.", "Document N0.", "Doc. No"]),
            "District": rng.choice(["Distrct", "Distric", "Dist."]),
            "Village": rng.choice(["Villge", "Vilage", "Vill."]),
        }
        for old, new in replacements.items():
            r.text = r.text.replace(old, new)

    return r


def tokenize(record: Record):
    # Exact whitespace tokenization keeps gold character spans stable.
    token_matches = list(re.finditer(r"\S+", record.text))
    tokens = [m.group(0) for m in token_matches]
    labels = ["O"] * len(tokens)

    for i, match in enumerate(token_matches):
        a, b = match.start(), match.end()
        overlapping = [
            (s, e, entity)
            for s, e, entity in record.spans
            if a < e and b > s
        ]

        if not overlapping:
            continue

        # Entity spans are intentionally space-separated in templates.
        s, e, entity = max(
            overlapping,
            key=lambda x: min(b, x[1]) - max(a, x[0]),
        )

        labels[i] = (
            f"B-{entity}" if a <= s else f"I-{entity}"
        )

    return {
        "tokens": tokens,
        "labels": labels,
        "text": record.text,
    }


def generate(n: int):
    return [tokenize(make_record()) for _ in range(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Separate RNG streams for deterministic, non-overlapping generation.
    global rng

    rng = random.Random(SEED)
    train = generate(TRAIN_SIZE)

    rng = random.Random(SEED + 1)
    validation = generate(VALIDATION_SIZE)

    rng = random.Random(SEED + 2)
    test = generate(TEST_SIZE)

    for filename, data in [
        ("train.json", train),
        ("validation.json", validation),
        ("test.json", test),
    ]:
        (OUT / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (OUT / "labels.json").write_text(
        json.dumps(
            {"labels": LABELS, "entity_types": ENTITY_TYPES},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Generated V4 dataset: {TRAIN_SIZE}/{VALIDATION_SIZE}/{TEST_SIZE}")
    print(f"Entity types: {len(ENTITY_TYPES)} | BIO labels: {len(LABELS)}")
    print(f"Output: {OUT.resolve()}")


if __name__ == "__main__":
    main()
