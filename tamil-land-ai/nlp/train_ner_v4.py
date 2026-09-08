"""Train IndicBERTv2 Tamil Land Record NER V4."""
import os,json,random,numpy as np,torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForTokenClassification,DataCollatorForTokenClassification,TrainingArguments,Trainer

MODEL_NAME="ai4bharat/IndicBERTv2-MLM-only"
BASE=Path(__file__).resolve().parents[1]
DATA=BASE/"data"/"ner_v4"; OUT=BASE/"models"/"land-ner-v4"
MAX_LENGTH=256; BATCH_SIZE=2; GRAD_ACC=8; EPOCHS=4; LR=2e-5; WD=.01; SEED=20260909
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load(p): return Dataset.from_list(json.loads(p.read_text(encoding="utf-8")))

def main():
    for p in [DATA/"train.json",DATA/"validation.json",DATA/"test.json",DATA/"labels.json"]:
        if not p.exists(): raise FileNotFoundError(f"Missing {p}. Run generate_dataset_v4.py first.")
    ld=json.loads((DATA/"labels.json").read_text(encoding="utf-8")); labels=ld["labels"]
    label2id={x:i for i,x in enumerate(labels)}; id2label={i:x for i,x in enumerate(labels)}
    train,val,test=load(DATA/"train.json"),load(DATA/"validation.json"),load(DATA/"test.json")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
    def align(examples):
        outs={"input_ids":[],"attention_mask":[],"labels":[]}
        for toks,labs in zip(examples["tokens"],examples["labels"]):
            enc=tokenizer(toks,is_split_into_words=True,truncation=True,max_length=MAX_LENGTH,padding=False)
            wids=enc.word_ids(); aligned=[]; prev=None
            for wid in wids:
                if wid is None: aligned.append(-100); continue
                lab=labs[wid]
                if wid!=prev: aligned.append(label2id[lab])
                else:
                    aligned.append(label2id.get("I-"+lab[2:],label2id[lab]) if lab.startswith("B-") else label2id[lab])
                prev=wid
            outs["input_ids"].append(enc["input_ids"]); outs["attention_mask"].append(enc["attention_mask"]); outs["labels"].append(aligned)
        return outs
    tr=train.map(align,batched=True,remove_columns=train.column_names,desc="Tokenizing train")
    va=val.map(align,batched=True,remove_columns=val.column_names,desc="Tokenizing validation")
    te=test.map(align,batched=True,remove_columns=test.column_names,desc="Tokenizing test")
    model=AutoModelForTokenClassification.from_pretrained(MODEL_NAME,num_labels=len(labels),id2label=id2label,label2id=label2id,trust_remote_code=True,ignore_mismatched_sizes=True)
    collator=DataCollatorForTokenClassification(tokenizer=tokenizer,padding=True,return_tensors="pt")
    def entities(seq):
        out=[]; typ=None; st=None
        for i,lid in enumerate(seq):
            lab=id2label[int(lid)]
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
        if typ: out.append((st,len(seq)-1,typ))
        return set(out)
    def metrics(pred):
        logits, labs=pred; preds=np.argmax(logits,axis=2); tp=pp=aa=0
        for pr,go in zip(preds,labs):
            p=[int(x) for x,y in zip(pr,go) if y!=-100]; g=[int(y) for y in go if y!=-100]
            pe,ge=entities(p),entities(g); tp+=len(pe&ge); pp+=len(pe); aa+=len(ge)
        precision=tp/pp if pp else 0.; recall=tp/aa if aa else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        return {"precision":precision,"recall":recall,"f1":f1}
    args=TrainingArguments(output_dir=str(OUT),num_train_epochs=EPOCHS,per_device_train_batch_size=BATCH_SIZE,per_device_eval_batch_size=BATCH_SIZE,gradient_accumulation_steps=GRAD_ACC,learning_rate=LR,weight_decay=WD,evaluation_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="f1",greater_is_better=True,logging_steps=50,save_total_limit=2,fp16=torch.cuda.is_available(),report_to="none",seed=SEED)
    trainer=Trainer(model=model,args=args,train_dataset=tr,eval_dataset=va,data_collator=collator,compute_metrics=metrics)
    print(f"Training examples: {len(train)} | labels: {len(labels)} | device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    trainer.train(); print("Validation:",trainer.evaluate(va)); print("Test:",trainer.evaluate(te))
    trainer.save_model(str(OUT)); tokenizer.save_pretrained(str(OUT)); (OUT/"labels.json").write_text(json.dumps(ld,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved V4 model to {OUT}")
if __name__=="__main__": main()
