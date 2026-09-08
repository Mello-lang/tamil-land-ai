"""Generate V4 Tamil/Tamil-English land-record NER data.

V4 goals:
- Preserve V3 core entities.
- Add DATE, OFFICER, HUSBAND_NAME, MOTHER_NAME, WIFE_NAME,
  RELATIVE_NAME, RELATION, JOINT_OWNER, APPLICANT, SELLER, BUYER.
- Add old/current survey and patta identifiers, subdivision, registration number.
- Add bilingual/mixed Tamil-English templates and realistic identifier variants.
- Add boundary-direction, tax, transaction, crop/irrigation and legal vocabulary.

The generator annotates exact character spans, then tokenizes on whitespace/punctuation
without relying on a fragile post-hoc label search.
"""
from __future__ import annotations
import json, random, re
from pathlib import Path

SEED = 20260909
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 30000, 4000, 4000
OUT = Path("data") / "ner_v4"
random.seed(SEED)

ENTITY_TYPES = [
    "AREA","BOUNDARY","DISTRICT","DOCUMENT_NUMBER","FATHER_NAME","LAND_TYPE",
    "OWNER","PATTA_NUMBER","SURVEY_NUMBER","TALUK","VILLAGE","YEAR",
    "DATE","OFFICER","HUSBAND_NAME","MOTHER_NAME","WIFE_NAME","RELATIVE_NAME",
    "RELATION","JOINT_OWNER","APPLICANT","SELLER","BUYER","OLD_PATTA_NUMBER",
    "OLD_SURVEY_NUMBER","SUBDIVISION_NUMBER","TOWN_SURVEY_NUMBER","RE_SURVEY_NUMBER",
    "PLOT_NUMBER","LAYOUT_PLOT_NUMBER","DOOR_NUMBER","DOCUMENT_TYPE",
    "REGISTRATION_NUMBER","DOCUMENT_DATE","REGISTRATION_DATE","TRANSFER_DATE",
    "FINANCIAL_YEAR","LAND_CLASSIFICATION","LAND_USE","CROP","WATER_SOURCE",
    "GOVERNMENT_LAND","EXTENT_PURCHASED","LAND_TAX","WET_TAX","DRY_TAX",
    "OTHER_TAX","ACQUISITION_TYPE","PATTA_STATUS","CASE_NUMBER","COURT_CASE",
    "TENANT","LEASE","RIGHT_TYPE","REMARKS","ADDRESS","PINCODE"
]
LABELS=["O"]+[f"{p}-{t}" for t in ENTITY_TYPES for p in ("B","I")]

NAMES_TA=["ராமசாமி","முருகேசன்","செந்தில்குமார்","சுப்பிரமணியன்","பழனிச்சாமி","கண்ணன்","கோபால்","சிவக்குமார்","மணிகண்டன்","ராஜேந்திரன்","வெங்கடேசன்","சரவணன்","மோகன்","குமார்","சுரேஷ்","மகேந்திரன்","அருணாசலம்","துரைசாமி","நடராஜன்","ஜெயக்குமார்","பிரபாகரன்","விஜயகுமார்","தனபால்","சண்முகம்","கிருஷ்ணமூர்த்தி","ரமேஷ்","அன்பழகன்","சக்திவேல்","கார்த்திகேயன்","லட்சுமி","மீனாட்சி","கலையரசி"]
NAMES_EN=["Suresh Kumar","Rajendran","Murugan","Karthikeyan","Ramesh Kumar","Lakshmi Devi","Shanmugam","Senthil Kumar","Priya","Arun Kumar","Vijay Kumar","Meena"]
DIST_TA=["கோயம்புத்தூர்","திருச்சிராப்பள்ளி","மதுரை","திண்டுக்கல்","திருப்பூர்","ஈரோடு","சேலம்","கரூர்","நாமக்கல்","திருநெல்வேலி","தூத்துக்குடி","விருதுநகர்","தேனி","தஞ்சாவூர்","கடலூர்","செங்கல்பட்டு","திருவள்ளூர்","வேலூர்"]
DIST_EN=["Coimbatore","Tiruchirappalli","Madurai","Dindigul","Tiruppur","Erode","Salem","Karur","Namakkal","Tirunelveli","Thoothukudi","Virudhunagar","Theni","Thanjavur","Cuddalore","Chengalpattu","Tiruvallur","Vellore"]
TALUK_TA=["பொள்ளாச்சி","கோவை தெற்கு","கோவை வடக்கு","மேட்டுப்பாளையம்","சூலூர்","உடுமலைப்பேட்டை","வால்பாறை","திருப்பூர்","பல்லடம்","அவிநாசி","பழனி","திண்டுக்கல்","கரூர்","நாமக்கல்","சேலம்","ஈரோடு"]
TALUK_EN=["Pollachi","Coimbatore South","Coimbatore North","Mettupalayam","Sulur","Udumalpet","Valparai","Tiruppur","Palladam","Avinashi","Palani","Dindigul","Karur","Namakkal","Salem","Erode"]
VIL_TA=["ஆனைமலை","காளப்பட்டி","சோமந்துறை","புத்தாநத்தம்","செட்டிபாளையம்","கிணத்துக்கடவு","மதுக்கரை","சுல்தான்பேட்டை","கோவில்பாளையம்","வடக்கலூர்","வெள்ளலூர்","சரவணம்பட்டி","குனியமுத்தூர்","பேரூர்","நாச்சிபாளையம்","வால்பாறை","உடுமலை","குரும்பபாளையம்","துடியலூர்","அன்னூர்","மேட்டுப்பாளையம்","சிங்காநல்லூர்","சூலூர்"]
VIL_EN=["Kinathukadavu","Kalapatti","Anaimalai","Pollachi","Madukkarai","Sulur","Annur","Perur","Kovilpalayam","Saravanampatti","Vellalore","Thudiyalur","Mettupalayam","Karamadai"]
LAND_TA=["நஞ்சை","புஞ்சை","மானாவாரி நிலம்","நத்தம்","தோட்ட நிலம்"]
LAND_EN=["wet land","dry land","agricultural land","residential land","garden land"]
CROPS_TA=["நெல்","தென்னை","கரும்பு","வாழை","மக்காச்சோளம்","காய்கறி"]
CROPS_EN=["paddy","coconut","sugarcane","banana","maize","vegetables"]
WATER_TA=["கிணறு","கால்வாய்","வாய்க்கால்","ஆறு","ஏரி","பாசன கால்வாய்"]
WATER_EN=["well","irrigation channel","canal","river","lake","irrigation canal"]


def val(name):
    pools={"owner":NAMES_TA+NAMES_EN,"district":DIST_TA+DIST_EN,"taluk":TALUK_TA+TALUK_EN,"village":VIL_TA+VIL_EN,"land":LAND_TA+LAND_EN,"crop":CROPS_TA+CROPS_EN,"water":WATER_TA+WATER_EN}
    return random.choice(pools[name])

def date(): return f"{random.randint(1,28):02d}-{random.randint(1,12):02d}-{random.randint(2000,2026)}"
def year(): return str(random.randint(1950,2026))
def fy():
    y=random.randint(2000,2026); return f"{y}-{str((y+1)%100).zfill(2)}"
def survey():
    n=random.randint(1,999); s=random.randint(1,30); suffix=random.choice(["","A","B","C","A1","B1","A2","B2","-1"])
    return f"{n}/{s}{suffix}"
def patta(): return str(random.randint(100000,999999))
def doc():
    return random.choice([str(random.randint(100000,999999)),f"DOC/{random.choice(['CBR','MDU','TPR','SLM'])}/{random.randint(2010,2026)}/{random.randint(1,99999):05d}",f"{random.randint(1,9999)}/{random.randint(2010,2026)}",f"DOC-{random.randint(1,99)}/{random.randint(2010,2026)}"])
def area():
    return random.choice([f"{random.randint(1,9)}.{random.randint(1,99):02d} acres",f"{random.randint(1,9)} acres",f"{random.randint(10,99)} cents",f"{random.randint(1,9)} acres {random.randint(1,99)} cents",f"{random.randint(500,20000)} sq ft",f"{random.randint(1,8)}.{random.randint(1,99):02d} ஏக்கர்",f"{random.randint(10,99)} சென்ட்"])

def number_unit(): return f"{random.randint(100,99999):.0f}"

class Rec:
    def __init__(self): self.text=""; self.spans=[]
    def add(self, s, typ=None):
        if self.text and not self.text.endswith((" ","\n")): self.text+=" "
        start=len(self.text); self.text+=s; end=len(self.text)
        if typ: self.spans.append((start,end,typ))
    def line(self, *parts):
        if self.text: self.text+="\n"
        for i,(s,t) in enumerate(parts):
            if i: self.text+=" "
            st=len(self.text); self.text+=s
            if t: self.spans.append((st,len(self.text),t))

def make_record():
    r=Rec(); owner=random.choice(NAMES_TA+NAMES_EN); father=random.choice(NAMES_TA+NAMES_EN); husband=random.choice(NAMES_TA+NAMES_EN); officer=random.choice(NAMES_TA+NAMES_EN)
    district=random.choice(DIST_TA+DIST_EN); taluk=random.choice(TALUK_TA+TALUK_EN); village=random.choice(VIL_TA+VIL_EN)
    s=survey(); sub=s.split('/',1)[1]; p=patta(); oldp=patta(); olds=survey(); d=doc(); dt=date(); y=year(); ar=area(); crop=val('crop'); water=val('water'); land=val('land')
    style=random.choices(["ta","en","mix"],[.45,.20,.35])[0]
    if style=="ta":
        r.line(("உரிமையாளர்",None),(owner,"OWNER"),("தந்தை பெயர்",None),(father,"FATHER_NAME"))
        r.line(("மாவட்டம்",None),(district,"DISTRICT"),("வட்டம்",None),(taluk,"TALUK"),("கிராமம்",None),(village,"VILLAGE"))
        r.line(("பட்டா எண்",None),(p,"PATTA_NUMBER"),("சர்வே எண்",None),(s,"SURVEY_NUMBER"),("உட்பிரிவு எண்",None),(sub,"SUBDIVISION_NUMBER"))
        r.line(("பழைய பட்டா எண்",None),(oldp,"OLD_PATTA_NUMBER"),("பழைய சர்வே எண்",None),(olds,"OLD_SURVEY_NUMBER"))
        r.line(("ஆவண எண்",None),(d,"DOCUMENT_NUMBER"),("ஆவண தேதி",None),(dt,"DOCUMENT_DATE"))
        r.line(("பதிவு ஆண்டு",None),(y,"YEAR"),("தேதி",None),(dt,"DATE"),("நிதியாண்டு",None),(fy(),"FINANCIAL_YEAR"))
        r.line(("பரப்பளவு",None),(ar,"AREA"),("நில வகை",None),(land,"LAND_TYPE"),("பயிர்",None),(crop,"CROP"))
        r.line(("நீராதாரம்",None),(water,"WATER_SOURCE"),("கண்காணிப்பு அதிகாரி",None),(officer,"OFFICER"))
    elif style=="en":
        r.line(("Owner",None),(owner,"OWNER"),("Father Name",None),(father,"FATHER_NAME"))
        r.line(("District",None),(district,"DISTRICT"),("Taluk",None),(taluk,"TALUK"),("Village",None),(village,"VILLAGE"))
        r.line(("Patta Number",None),(p,"PATTA_NUMBER"),("Survey No.",None),(s,"SURVEY_NUMBER"),("Subdivision No.",None),(sub,"SUBDIVISION_NUMBER"))
        r.line(("Old Patta No.",None),(oldp,"OLD_PATTA_NUMBER"),("Old Survey No.",None),(olds,"OLD_SURVEY_NUMBER"))
        r.line(("Document No.",None),(d,"DOCUMENT_NUMBER"),("Document Date",None),(dt,"DOCUMENT_DATE"))
        r.line(("Registration Year",None),(y,"YEAR"),("Date",None),(dt,"DATE"),("Financial Year",None),(fy(),"FINANCIAL_YEAR"))
        r.line(("Extent",None),(ar,"AREA"),("Land Type",None),(land,"LAND_TYPE"),("Crop",None),(crop,"CROP"))
        r.line(("Water Source",None),(water,"WATER_SOURCE"),("Verified by Officer",None),(officer,"OFFICER"))
    else:
        r.line(("உரிமையாளர் / Owner",None),(owner,"OWNER"),("Father Name / தந்தை பெயர்",None),(father,"FATHER_NAME"))
        r.line(("District / மாவட்டம்",None),(district,"DISTRICT"),("Taluk / வட்டம்",None),(taluk,"TALUK"),("Village / கிராமம்",None),(village,"VILLAGE"))
        r.line(("Patta No.",None),(p,"PATTA_NUMBER"),("Survey No.",None),(s,"SURVEY_NUMBER"),("Sub Division",None),(sub,"SUBDIVISION_NUMBER"))
        r.line(("Old Patta No.",None),(oldp,"OLD_PATTA_NUMBER"),("Old Survey No.",None),(olds,"OLD_SURVEY_NUMBER"))
        r.line(("Document No.",None),(d,"DOCUMENT_NUMBER"),("ஆவண தேதி / Document Date",None),(dt,"DOCUMENT_DATE"))
        r.line(("Registration Year",None),(y,"YEAR"),("தேதி / Date",None),(dt,"DATE"),("Financial Year",None),(fy(),"FINANCIAL_YEAR"))
        r.line(("Extent / பரப்பளவு",None),(ar,"AREA"),("Land Type / நில வகை",None),(land,"LAND_TYPE"),("Crop / பயிர்",None),(crop,"CROP"))
        r.line(("Water Source / நீராதாரம்",None),(water,"WATER_SOURCE"),("Verified by Officer / அதிகாரி",None),(officer,"OFFICER"))
    # Add secondary sections with varied contexts.
    r.line(("North",None),(random.choice(NAMES_TA+NAMES_EN)+" land","BOUNDARY"),("South",None),(random.choice(["Main Road","முக்கிய சாலை","canal","கால்வாய்"]),"BOUNDARY"))
    r.line(("East",None),(random.choice(["Government Poramboke land","அரசு புறம்போக்கு நிலம்","Public Road"]),"GOVERNMENT_LAND"),("West",None),(random.choice(["irrigation channel","வாய்க்கால்","Main Road"]),"BOUNDARY"))
    r.line(("Sale Deed", "DOCUMENT_TYPE"),(d,"DOCUMENT_NUMBER"),("Registration No.",None),(doc(),"REGISTRATION_NUMBER"))
    r.line(("Sale Date",None),(dt,"SALE_DEED_DATE"),("Transfer Date",None),(date(),"TRANSFER_DATE"),("Purchased",None),(area(),"EXTENT_PURCHASED"))
    r.line(("Seller",None),(random.choice(NAMES_TA+NAMES_EN),"SELLER"),("Buyer",None),(owner,"BUYER"))
    r.line(("Applicant",None),(owner,"APPLICANT"),("Relation",None),(random.choice(["son of","daughter of","த/பெ","கணவர்","wife of"]),"RELATION"),("Relative",None),(father,"RELATIVE_NAME"))
    r.line(("Land Tax",None),(number_unit(),"LAND_TAX"),("Wet Tax",None),(number_unit(),"WET_TAX"),("Dry Tax",None),(number_unit(),"DRY_TAX"))
    r.line(("Acquisition",None),(random.choice(["Sale","Inheritance","Gift","Settlement","Family Partition","விற்பனை","வாரிசு"]),"ACQUISITION_TYPE"),("Status",None),(random.choice(["Active","Transferred","தீர்வு செய்யப்பட்டது"]),"PATTA_STATUS"))
    r.line(("Door No.",None),(f"{random.randint(1,999)}/{random.choice('ABCDEFG')}","DOOR_NUMBER"),("Plot No.",None),(str(random.randint(1,999)),"PLOT_NUMBER"),("Pincode",None),(f"{random.randint(600000,643999)}","PINCODE"))
    # Hard negatives: numbers/IDs in non-target contexts.
    r.line(("Page No.",None),(str(random.randint(1,250)),None),("File Ref.",None),(f"FILE-{random.randint(1000,9999)}",None),("Phone",None),(f"9{random.randint(100000000,999999999)}",None))
    return r

def tokenize(rec:Rec):
    # Whitespace tokens; punctuation stays attached to preserve exact source spans.
    # Each token gets a label based on overlap with a gold span.
    matches=[]
    for m in re.finditer(r"\S+", rec.text): matches.append((m.start(),m.end(),m.group()))
    labels=[]
    for a,b,tok in matches:
        hits=[(e-s,s,e,typ) for s,e,typ in rec.spans if a<e and b>s]
        if not hits: labels.append("O"); continue
        # Prefer the most specific/largest overlap; spans are non-overlapping by construction.
        _,s,e,typ=max(hits,key=lambda x:x[0])
        labels.append(("B-" if a<=s else "I-")+typ)
    return {"tokens":[x[2] for x in matches],"labels":labels,"text":rec.text}

def generate(n):
    return [tokenize(make_record()) for _ in range(n)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Fixed split generation keeps reproducibility and avoids cross-run drift.
    train, val, test=generate(TRAIN_SIZE),generate(VAL_SIZE),generate(TEST_SIZE)
    for name,data in [("train.json",train),("validation.json",val),("test.json",test)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"labels.json").write_text(json.dumps({"labels":LABELS,"entity_types":ENTITY_TYPES},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Generated {TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE} examples in {OUT}")
    print(f"Entity types: {len(ENTITY_TYPES)} | BIO labels: {len(LABELS)}")

if __name__=="__main__": main()
