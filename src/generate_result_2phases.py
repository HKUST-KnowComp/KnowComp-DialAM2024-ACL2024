from cgi import test
import json
import argparse
import torch
import os
import time
import datetime

from transformers import  AutoModelForSequenceClassification,AutoTokenizer

from tqdm import tqdm

    
RELATION_MAP_P1 = {
    "yes":0,
    "no":1,
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
RELATION_P1 = ["yes","no"]
RELATION_P2 = ["RA","CA","MA"]
RELATION_TEXT = ["Default Inference","Default Conflict","Default Rephrase"]
YA_RELATION = ['Asserting','Restating','Arguing','Pure Questioning','Default Illocuting','Disagreeing','Agreeing','Assertive Questioning','Rhetorical Questioning','Challenging','Analysing','None']

def main(ckpt_dir,phase=1,phase_test_dir=None,phase_res_dir=None):
    # time1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # print(f"{time1}: start generating results with ckpt {ckpt_dir}")
    gen_batch_size = 256
    ckpt_start_time = time.perf_counter()
    num_labels = 2
    if phase == 2:
        num_labels = 3
    if "deberta" in ckpt_dir:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir,num_labels=num_labels,device_map="cuda")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir,num_labels=num_labels,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    
    test_res_dir = os.path.join(ckpt_dir,"pred_results")
    test_dir = "./dataset/test/my_test/"
    if phase_test_dir:
        test_dir = phase_test_dir
    if phase_res_dir:
        test_res_dir = phase_res_dir
    if not os.path.exists(test_res_dir):
        os.makedirs(test_res_dir)
        
    for test_file in os.listdir(test_dir):
        
        file_time_start = time.perf_counter()

        test_path = os.path.join(test_dir,test_file)
        test_res_path = os.path.join(test_res_dir,test_file)
        
        with open(test_path,"r",encoding="utf-8") as f:
            data = json.loads(f.read())
        
        inodes = []
        node_maps = {}
        node_id_start = 0
        for node in data["nodes"]:
            node_id_start = max(int(node["nodeID"]),node_id_start)
            node_maps[node["nodeID"]] = node
            if node["type"] == "I":
                inodes.append(node)
        node_id_start += 1
        
        edge_id_start = 0
        for edge in data["edges"]:
            edge_id_start = max(int(edge["edgeID"]),edge_id_start)
        edge_id_start += 1
        
        input_text_batch = []
        input_text_nodes = []
        tmp_batch = []
        tmp_node_batch = []

        if phase == 1:
            for i in range(len(inodes)):
                for j in range(len(inodes)):
                    a = inodes[i]
                    b = inodes[j]
                    if a==b:
                        continue
                    input_text = "HEAD:{}  TAIL:{}  HAVING RELATION:A. yes, B. no".format(a["text"],b["text"])           
                    tmp_batch.append(input_text)
                    tmp_node_batch.append([a,b])
                    if len(tmp_batch) >= gen_batch_size:
                        input_text_batch.append(tmp_batch[:])
                        input_text_nodes.append(tmp_node_batch[:])
                        tmp_batch = list()
                        tmp_node_batch = list()
            if len(tmp_batch)>0:
                input_text_batch.append(tmp_batch)
                input_text_nodes.append(tmp_node_batch)
            
            
            
            for i in range(len(input_text_batch)): 
                batch = input_text_batch[i]
                node_batch = input_text_nodes[i]
                inputs = tokenizer(batch,return_tensors="pt",truncation=True,padding=True).to("cuda")
                
                with torch.no_grad():
                    logits = model(**inputs).logits.cpu()
                
                for j in range(len(batch)):
                    predicted_class_id = logits[j].argmax().item()
                    predicted_relation = RELATION_P1[predicted_class_id]
                    if predicted_relation == "no":
                        # print("1232132131232131")
                        continue
                    # output_text = tokenizer.decode(outputs)        
                    new_node = {"nodeID":str(node_id_start),"type":"HAVING RELATION","node_id":{"head":str(node_batch[j][0]["nodeID"]),"tail":str(node_batch[j][1]["nodeID"])}}
                        
                    new_edge_in ={"edgeID":str(edge_id_start),"fromID":str(node_batch[j][0]["nodeID"]),"toID":str(node_id_start)}
                    new_edge_out = {"edgeID":str(edge_id_start+1),"fromID":str(node_id_start),"toID":str(node_batch[j][1]["nodeID"])}
                    node_id_start += 1
                    edge_id_start += 2
                    data["nodes"].append(new_node)
                    data["edges"].extend([new_edge_in,new_edge_out])    
        
        elif phase == 2:
            new_nodes = []
            for a in data["nodes"]:
                if not a["type"] == "HAVING RELATION":
                    new_nodes.append(a)
                    continue
                input_text = "HEAD:{}  TAIL:{}  RELATION:A. RA; B. CA; C. MA".format(node_maps[a["node_id"]["head"]]["text"],node_maps[a["node_id"]["tail"]]["text"])
                tmp_batch.append(input_text)
                tmp_node_batch.append(a)
                if len(tmp_batch) >= gen_batch_size:
                    input_text_batch.append(tmp_batch[:])
                    input_text_nodes.append(tmp_node_batch[:])
                    tmp_batch = list()
                    tmp_node_batch = list()
            if len(tmp_batch)>0:
                input_text_batch.append(tmp_batch)
                input_text_nodes.append(tmp_node_batch)
            
            
            for i in range(len(input_text_batch)): 
                batch = input_text_batch[i]
                node_batch = input_text_nodes[i]
                inputs = tokenizer(batch,return_tensors="pt",truncation=True,padding=True).to("cuda")
                
                with torch.no_grad():
                    logits = model(**inputs).logits.cpu()
                
                for j in range(len(batch)):
                    predicted_class_id = logits[j].argmax().item()
                    if predicted_class_id == 3:
                        continue
                    predicted_relation = RELATION_P2[predicted_class_id]

                    # output_text = tokenizer.decode(outputs)        
                    new_node = {"nodeID":str(node_batch[j]["nodeID"]),"type":predicted_relation,"text":RELATION_TEXT[predicted_class_id],"scheme":RELATION_TEXT[predicted_class_id]}                  
                    new_nodes.append(new_node)
            data["nodes"] = new_nodes
            
            
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
    parser.add_argument("--model_dir",type=str,default=None)
    parser.add_argument("--phase",type=int,default=1)
    parser.add_argument("--phase_data",type=str,default=None)
    parser.add_argument("--phase_res",type=str,default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # for ckpt_dir in os.listdir(args.model_dir):
        # for ckpt_dir in [f"checkpoint-{x}" for x in [485]]:
    ckpt_dir = "./output/relation_models_p2/mnli-robert5e-6"
    ckpt_dir = args.model_dir
        # main(os.path.join(args.model_dir,ckpt_dir),args.phase,args.phase_data)
    main(ckpt_dir,args.phase,args.phase_data,args.phase_res) 
# 