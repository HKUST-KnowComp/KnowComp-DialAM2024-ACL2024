import json
import argparse
import torch
import os
import time
import datetime

from transformers import DebertaV2ForSequenceClassification, DebertaTokenizer, AutoModelForSequenceClassification,AutoTokenizer

RELATION_MAP = {
    "RA":0,
    "CA":1,
    "MA":2,
    "None":3
}

YA_RELATION_MAP={
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
RELATION = ["RA","CA","MA","None"]
RELATION_TEXT = ["Default Inference","Default Conflict","Default Rephrase"]
YA_RELATION = ['Asserting','Restating','Arguing','Pure Questioning','Default Illocuting','Disagreeing','Agreeing','Assertive Questioning','Rhetorical Questioning','Challenging','Analysing','None']

def main(ckpt_dir,batch_size,is_ya,ppl_test_dir=None,ppl_test_res_dir=None):

    ckpt_start_time = time.perf_counter()
    num_labels = 4
    if is_ya:
        num_labels = 12
    if "deberta" in ckpt_dir:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir,num_labels=num_labels,device_map="cuda")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir,num_labels=num_labels,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    test_res_dir = os.path.join(ckpt_dir,"pred_results")
    if is_ya:
        test_dir = "./dataset/test/my_test_ya/"
    else:
        test_dir = "./dataset/test/my_test/"
    if ppl_test_dir:
        test_dir = ppl_test_dir
    if ppl_test_res_dir:
        test_res_dir = ppl_test_res_dir

    if not os.path.exists(test_res_dir):
        os.makedirs(test_res_dir)
        
    for test_file in os.listdir(test_dir):
        
        file_time_start = time.perf_counter()

        test_path = os.path.join(test_dir,test_file)
        test_res_path = os.path.join(test_res_dir,test_file)
        
        with open(test_path,"r",encoding="utf-8") as f:
            data = json.loads(f.read())
        
        head_nodes = []
        tail_nodes = []
        
        node_id_start = 0
        node_map = {}
        for node in data["nodes"]:
            node_id_start = max(int(node["nodeID"]),node_id_start)
            node_map[node["nodeID"]] = node
        node_id_start += 1
        for node in data["nodes"]:
            if node["type"] == "L":
                head_nodes.append(node)
            elif node["type"] == "I":
                tail_nodes.append(node)
            
        edge_id_start = 0
        for edge in data["edges"]:
            edge_id_start = max(int(edge["edgeID"]),edge_id_start)
        edge_id_start += 1
        
        
        for a in tail_nodes:
            input_text_batch = []
            input_text_nodes = []
            target_node = None
            target_score = 0
            target_ya_class_id = None
            for b in head_nodes:
                input_texts = ["HEAD:"]
                input_texts.append("{}.{}".format(1,b["text"]))    
                input_texts.append("  TAIL:")
                input_texts.append("{}.{}".format(1,a["text"]))
                input_texts.append("  RELATION:")

                input_text = ''.join(input_texts)
                input_text_batch.append(input_text)
                input_text_nodes.append(b)

            i = 0
            while i<len(input_text_batch):
                if len(input_text_batch) - i < batch_size:
                    batch = input_text_batch[i:]
                    node_batch = input_text_nodes[i:]
                else:
                    batch = input_text_batch[i:i+batch_size]
                    node_batch = input_text_nodes[i:i+batch_size]
                i += batch_size
                    
                inputs = tokenizer(batch,return_tensors="pt",truncation=True,padding=True).to("cuda")
                
                with torch.no_grad():
                    logits = model(**inputs).logits.cpu()
                
                for j in range(len(batch)):
                    predicted_class_id = logits[j].argmax().item()
                    predicted_ya_score = logits[j][predicted_class_id]
                    if predicted_ya_score < target_score or predicted_class_id == 11:
                        continue
                    
                    target_node = node_batch[j]
                    target_score = predicted_ya_score
                    target_ya_class_id = predicted_class_id
            
                    
                    predicted_relation = YA_RELATION[target_ya_class_id]
                    if predicted_relation == "None":
                        continue
            
                    new_node = {"nodeID":str(node_id_start),"text":predicted_relation,"type":"YA","scheme":predicted_relation}
                    new_edge_in ={"edgeID":str(edge_id_start),"fromID":target_node["nodeID"],"toID":str(node_id_start)}
                    new_edge_out = {"edgeID":str(edge_id_start+1),"fromID":str(node_id_start),"toID":a["nodeID"]}
                    node_id_start += 1
                    edge_id_start += 2
                    data["nodes"].append(new_node)
                    data["edges"].extend([new_edge_in,new_edge_out])
                

            
            
        with open(test_res_path,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=4)
        
        file_time_end = time.perf_counter()
        time1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("{}: {:.2f}s on file {}".format(time1,file_time_end-file_time_start,test_file))
    
    ckpt_end_time = time.perf_counter()
    time1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("{}: {:.2f}s on ckpt {}".format(time1,ckpt_end_time-ckpt_start_time,ckpt_dir))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("--is_ya",type=bool,default=True)
    parser.add_argument("--test_dir",type=str,default=None) # path of the data to be predicted
    parser.add_argument("--res_dir",type=str,default=None) # path for saving the results
    parser.add_argument("--batch_sz",type=int,default=256) 
    args = parser.parse_args()
    

    ckpt_dir = args.model_path
    main(ckpt_dir,batch_size=args.batch_sz,is_ya=args.is_ya,ppl_test_dir=args.test_dir,ppl_test_res_dir=args.res_dir)
