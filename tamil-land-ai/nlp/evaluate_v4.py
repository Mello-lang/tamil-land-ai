"""Evaluate exact-span entity F1 for a V4 JSON split."""
import json,sys
from pathlib import Path
import torch,numpy as np
from transformers import AutoTokenizer,AutoModelForTokenClassification
BASE=Path(__file__).resolve().parents[1]; MODEL=BASE/"models"/"land-ner-v4"; DATA=BASE/"data"/"ner_v4"

def ents(labels,id2):
    out=[]; typ=None; st=None
    for i,lid in enumerate(labels):
        lab=id2[int(lid)]
        if lab=="O":
            if typ: out.append((st,i-1,typ)); typ=st=None
        elif lab.startswith("B-"):
            if typ: out.append((st,i-1,typ))
            typ=lab[2:]; st=i
        else:
            et=lab[2:]
            if typ!=et:
                if typ: out.append((st,i-1,typ))
                typ=et; st=i
    if typ: out.append((st,len(labels)-1,typ))
    return set(out)

def main():
    split=sys.argv[1] if len(sys.argv)>1 else "test"; data=json.loads((DATA/f"{split}.json").read_text(encoding="utf-8")); tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True); model=AutoModelForTokenClassification.from_pretrained(str(MODEL),trust_remote_code=True)
    id2=model.config.id2label; counts={}; total=[0,0,0]
    for ex in data:
        enc=tok(ex["tokens"],is_split_into_words=True,truncation=True,max_length=256); w=enc.word_ids(); x=model(**{k:torch.tensor([v]) for k,v in enc.items()}); pred=x.logits.argmax(-1)[0].tolist(); gold=[]; prev=None
        for wid in w:
            if wid is None: gold.append(-100)
            else: gold.append(model.config.label2id[ex["labels"][wid]])
        p=[a for a,b in zip(pred,gold) if b!=-100]; g=[b for b in gold if b!=-100]
        pe,ge=ents(p,id2),ents(g,id2)
        for _,_,t in ge: counts.setdefault(t,[0,0,0])[2]+=1
        for _,_,t in pe: counts.setdefault(t,[0,0,0])[1]+=1
        for e in pe&ge: counts.setdefault(e[2],[0,0,0])[0]+=1
    print("type\tprecision\trecall\tf1\tTP\tPred\tGold")
    for t,c in sorted(counts.items()):
        tp,pp,gg=c; pr=tp/pp if pp else 0; rc=tp/gg if gg else 0; f=2*pr*rc/(pr+rc) if pr+rc else 0
        print(f"{t}\t{pr:.4f}\t{rc:.4f}\t{f:.4f}\t{tp}\t{pp}\t{gg}")
if __name__=="__main__": main()
