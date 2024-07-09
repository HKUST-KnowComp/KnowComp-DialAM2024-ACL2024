import json
import os
import argparse
import random


YA_RELATION_MAP={'Asserting': 19258, 
 'Restating': 4099, 
 'Arguing': 5505, 
 'Pure Questioning': 1203, 
 'Default Illocuting': 1922, 
 'Disagreeing': 1252, 
 'Agreeing': 342, 
 'Assertive Questioning': 245, 
 'Rhetorical Questioning': 224, 
 'Challenging': 124, 
 'Analysing': 256}

def extrac_eval(file):
    with open(f"./dataset/train/train_data/{file}","r",encoding="utf-8") as f:
        data = json.loads(f.read())

    new_nodes = []
    ya_node_ids = []
    for node in data["nodes"]:
        if node["type"] == "YA":
            ya_node_ids.append(node["nodeID"])
            continue
        new_nodes.append(node)
            
    new_edges = []
    for edge in data["edges"]:
        if edge["fromID"] in ya_node_ids or edge["toID"] in ya_node_ids:
            continue
        new_edges.append(edge)
        
    data["nodes"] = new_nodes
    data["edges"] = new_edges
    
    with open(f"./dataset/test/my_test_ya/{file}","w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
        
        
def getnodes(edges,node,node_type,node_maps):
    inbounds,outbounds = [],[]
    for edge in edges:
        if edge["fromID"] == node["nodeID"]:# outbound edge
            out_node = node_maps[edge['toID']]
            if out_node["type"] == node_type:# only consider I-nodes
                outbounds.append(out_node) # add the node from another end
                
        elif edge["toID"] == node["nodeID"]:# inbound edge
            in_node = node_maps[edge['fromID']]
            if in_node["type"] == node_type:
                inbounds.append(in_node) # add the node from another end
    return inbounds,outbounds


def get_relation_tuples(relation_nodes,edges,node_maps):
    res = []
    for node in relation_nodes:
        inbounds = []
        outbounds = []
        for edge in edges:
            if edge["fromID"] == node["nodeID"]:# outbound edge
                out_node = node_maps[edge['toID']]
                if out_node["type"] == "I":# only consider I-nodes
                    outbounds.append(out_node) # add the node from another end
                elif out_node["type"] in ["RA","CA","MA"]:
                    sub_in, sub_out = getnodes(edges,out_node,"I",node_maps)
                    outbounds.extend(sub_in+sub_out)
            elif edge["toID"] == node["nodeID"]:# inbound edge
                in_node = node_maps[edge['fromID']]
                if in_node["type"] == "L":
                    inbounds.append(in_node) # add the node from another end
                elif in_node["type"] == "TA":
                    sub_in,sub_out = getnodes(edges,in_node,"L",node_maps)
                    inbounds.extend(sub_in+sub_out)
        # if len(inbounds) > 1:
        #     print("multiple inbound edges in nodeID:{} type:{}".format(node["nodeID"],node["type"]))
        # if len(outbounds) > 1:
        #     print("multiple outbound edges in nodeID:{} type:{}".format(node["nodeID"],node["type"]))
        res.append({"head":inbounds,"relation":node,"tail":outbounds})
    return res

def get_new_data(data_path,neg=1):
    with open(data_path,"r",encoding="utf-8") as f:
        data = f.read()
        data = json.loads(data)
    nodes = data["nodes"]
    edges = data["edges"]
    
    # get node
    ya_nodes = []
    tmp_node_map = {}
    for node in nodes:
        tmp_node_map.update({node["nodeID"]:node})
        if node["type"] == "YA":
            ya_nodes.append(node)

    # get inbound and outbound edges
    res_YA = get_relation_tuples(ya_nodes,edges,tmp_node_map)

    
    res_No = []
    
    def checknode(node1,node2,res_ya):
        for x in res_ya :
            if node1 in x["head"] and node2 in x["tail"]:
                return False
            elif node1 in x["tail"] and node2 in x["head"]:
                return False
        if node1["type"] in ["L","TA"] and node2["type"] in ["I","RA","CA","MA"]:
            return True
        if node2["type"] in ["L","TA"] and node1["type"] in ["I","RA","CA","MA"]:
            return True
        return False
    
    neg_num = int(len(res_YA)*neg)
    for _ in range(neg_num): #  add neg samples
        while True:
            new_idx_1 = random.randint(0,len(nodes)-1)
            new_idx_2 = random.randint(0,len(nodes)-1)
            new_node1, new_node2 = nodes[new_idx_1],nodes[new_idx_2]
            if new_node1 not in ya_nodes and new_node2 not in ya_nodes and checknode(new_node1,new_node2,res_YA):
                res_No.append({"head":[new_node1],"relation":{"type":"None","text":"None"},"tail":[new_node2]})
                break
    
    return res_YA, res_No

def process(data):
    processed_data = []
    for triple in data:
        input_texts = ["HEAD:"]
        for i in range(len(triple["head"])):
            input_texts.append("{}.{}".format(i+1,triple["head"][i]["text"]))
        input_texts.append("  TAIL:")
        for i in range(len(triple["tail"])):
            input_texts.append("{}.{}".format(i+1,triple["tail"][i]["text"]))    
        input_texts.append("  RELATION:")
        # input_texts.append("A. RA; B. CA; C. MA; D. None")
        input_text = ''.join(input_texts)
        processed_data.append({"text":input_text,"labels":triple["relation"]["text"]})
    return processed_data


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str)
    parser.add_argument("--data_path",type=str,default=None)
    parser.add_argument("--save_dir",type=str)
    parser.add_argument("--eval_rate",type=float,default=0.05) # the proportion of eval data 
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--neg",type=float,default=2) # the num of added neg samples : the num of original training samples
    args = parser.parse_args()
    new_ya,new_no = [],[]
    if not args.data_dir:
        tmp_data = get_new_data(args.data_path,neg=args.neg)
        new_ya += tmp_data[0]
        new_no += tmp_data[1]
    else:
        datafiles = os.listdir(args.data_dir)
        for subdata in datafiles:
            tmp_data = get_new_data(os.path.join(args.data_dir,subdata),neg=args.neg)
            new_ya += tmp_data[0]
            new_no += tmp_data[1]
            # extrac_eval(subdata)
        
    print(f"YA NUM:{len(new_ya)}  NONE NUM:{len(new_no)}")
    
    random.seed(42)
    
    random.shuffle(new_ya)
    random.shuffle(new_no)
    ya_eval_num = int(len(new_ya)*0.05)
    no_eval_num = int(len(new_no)*0.05)
    
    print(f"RA EVAL NUM:{ya_eval_num }  NONE EVAL NUM:{no_eval_num }")
    
    
    eval_data = new_ya[:ya_eval_num] + new_no[:no_eval_num]
    train_data = new_ya[ya_eval_num:] + new_no[no_eval_num:]
    random.shuffle(eval_data)
    random.shuffle(train_data)
    

    with open(os.path.join(args.save_dir,"train.json"),"w",encoding="utf-8") as saved_f:
        json.dump({"data":process(train_data)},saved_f,ensure_ascii=False,indent=4)
    with open(os.path.join(args.save_dir,"eval.json"),"w",encoding="utf-8") as saved_f:
        json.dump({"data":process(eval_data)},saved_f,ensure_ascii=False,indent=4)
        
