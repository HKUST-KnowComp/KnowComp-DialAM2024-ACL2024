import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

from transformers import TrainingArguments, Trainer, DebertaV2ForSequenceClassification, DebertaTokenizer, AutoModelForSequenceClassification,AutoTokenizer
from myDataset import RelationDataset
from datasets import load_dataset
from metrics import compute_metrics

RELATION_MAP={
 'Asserting': 0,#19258, 
 'Restating': 1,#4099, 
 'Arguing': 2,#5505, 
 'Pure Questioning': 3,#1203, 
 'Default Illocuting': 4,#1922, 
 'Disagreeing': 5,#1252, 
 'Agreeing': 6,#342, 
 'Assertive Questioning': 7,#245, 
 'Rhetorical Questioning': 8,#224, 
 'Challenging': 9,#124, 
 'Analysing': 10,#256}
 'None':11 #32656
}


def main():
    model = AutoModelForSequenceClassification.from_pretrained("/models/roberta-large-mnli",num_labels=12,ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained('/models/roberta-large-mnli')
    
    def preprocess(examples):
        model_inputs = tokenizer(examples["text"], truncation=True)
        labels = [RELATION_MAP[x] for x in examples["labels"]]
        model_inputs["labels"] = labels
        return model_inputs
    
    raw_datasets = load_dataset("./data/train/ya_relation_data_new/",data_files={"train":"train.json","eval":"eval.json"},field="data")
    relationDataset = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets['train'].column_names)
        
    training_args = TrainingArguments(
        output_dir="./output/ya_models/mnli-roberta-2e-5",
        learning_rate=2e-5,
        do_train=True,
        do_eval=True,
        fp16=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
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