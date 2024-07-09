import json
import argparse
import os

from transformers import TrainingArguments, Trainer, DebertaTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from metrics import compute_metrics

    
RELATION_MAP = {
    "yes":0,
    "no":1,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default="./models/deberta-v3-base",type=str)
    parser.add_argument("--data_path",default="./data/train/inode_relation_data_p1/",type=str)
    parser.add_argument("--output_dir",default="./output/relation_models_p1/deberta-base",type=str)
    parser.add_argument("--batch_sz",default=64,type=int)
    
    args = parser.parse_args()
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path,num_labels=2)
    tokenizer = DebertaTokenizer.from_pretrained(args.model_path)
    
    def preprocess(examples):
        model_inputs = tokenizer(examples["text"], truncation=True)
        labels = [RELATION_MAP[x] for x in examples["labels"]]
        model_inputs["labels"] = labels
        return model_inputs
    
    raw_datasets = load_dataset(args.data_path,data_files={"train":"train.json","eval":"eval.json"},field="data")
    relationDataset = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets['train'].column_names)
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=1e-5,
        do_train=True,
        do_eval=True,
        fp16=True,
        per_device_train_batch_size=args.batch_sz,
        per_device_eval_batch_size=args.batch_sz,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        log_level="info",
        # save_total_limit=2,
        save_safetensors= False,
        push_to_hub=False,
    )
    
      
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=relationDataset["train"],
            eval_dataset=relationDataset["eval"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__=="__main__":
    main()