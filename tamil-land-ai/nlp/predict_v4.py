"""Predict structured entities with V4."""
import json,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
BASE=Path(__file__).resolve().parents[1]; MODEL=BASE/"models"/"land-ner-v4"

def extract(text):
    tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True); model=AutoModelForTokenClassification.from_pretrained(str(MODEL),trust_remote_code=True)
    enc=tok(text,return_offsets_mapping=True,return_tensors="pt",truncation=True,max_length=256)
    offsets=enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad(): logits=model(**enc).logits[0]; probs=torch.softmax(logits,dim=-1); ids=logits.argmax(-1).tolist()
    ents=[]; cur=None
    for lid,(a,b) in zip(ids,offsets):
        if b<=a: continue
        lab=model.config.id2label[int(lid)]
        if lab=="O":
            if cur: ents.append(cur); cur=None
            continue
        typ=lab[2:] if lab.startswith(("B-","I-")) else lab; is_b=lab.startswith("B-")
        if is_b or cur is None or cur["type"]!=typ or a>cur["end"]:
            if cur: ents.append(cur)
            cur={"type":typ,"start":a,"end":b,"score":float(probs[lid].item())}
        else:
            cur["end"]=b; cur["score"]=(cur["score"]+float(probs[lid].item()))/2
    if cur: ents.append(cur)
    out={}
    for e in ents:
        val=text[e["start"]:e["end"]].strip(); out.setdefault(e["type"],[]).append({"text":val,"confidence":round(e["score"],4)})
    return out
if __name__=="__main__":
    text=" ".join(sys.argv[1:]) if len(sys.argv)>1 else input("Enter land-record text: ")
    print(json.dumps(extract(text),ensure_ascii=False,indent=2))
