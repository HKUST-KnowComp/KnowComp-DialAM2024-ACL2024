import json
import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

from transformers import TrainingArguments, Trainer, DebertaV2ForSequenceClassification, DebertaTokenizer, AutoModelForSequenceClassification, AutoModelForPreTraining, AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from metrics import compute_metrics

    
RELATION_MAP = {
    "RA":0,
    "CA":1,
    "MA":2,
}

def main():
    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained("/models/roberta-large-mnli",num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("/models/roberta-large-mnli")

    model.config.id2label = {0:"RA",1:"CA",2:"MA"}
    model.config.label2id = {"RA":0,"CA":1,"MA":2}

    def preprocess(examples):
        model_inputs = tokenizer(examples["text"], truncation=True)
        labels = [RELATION_MAP[x] for x in examples["labels"]]
        model_inputs["labels"] = labels
        return model_inputs
    
    raw_datasets = load_dataset("./data/train/inode_relation_data_p2/",data_files={"train":"train.json","eval":"eval.json"},field="data")
    relationDataset = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets['train'].column_names)
        
    training_args = TrainingArguments(
        output_dir="./output/relation_models_p2/mnli-robert5e-6",
        learning_rate=5e-6,
        do_train=True,
        do_eval=True,
        # fp16=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
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