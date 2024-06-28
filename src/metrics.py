import numpy as np
import re
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from transformers import EvalPrediction

def simple_accuracy(res):
    '''
    res[0]:prediction (output logits after softmax,  attention output)
    res[1]:labels 
    all np array 
    output logits shape: (batch_size, seq_len, num_vocab)
    labels shape: (batch_size, seq_len) 
    '''
    preds=res[0][0]   
    preds = np.argmax(preds,axis=2)
    labels=res[1]
    return {"accuracy":(preds == labels).mean().item()}

def multi_label_metrics(predictions, labels, threshold=0.5):
    probs =  np.argmax( predictions, -1)       
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=probs, average='micro')
    accuracy = accuracy_score(y_true, probs)
    # print(classification_report([id2label[x] for x in y_true], [id2label[x] for x in probs]))
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'accuracy': accuracy}
    return metrics
 
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result