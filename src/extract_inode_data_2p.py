import json
import argparse
import random
import os

RELATION_MAP = {
    "RA":0,
    "CA":1,
    "MA":2,
    "None":3
}

def get_relation_tuples(used_nodes,edges,node_maps):
    res = []
    for node in used_nodes:
        inbounds = []
        outbounds = []
        for edge in edges:
            if edge["fromID"] == node["nodeID"]:# outbound edge
                out_node = node_maps[edge['toID']]
                if out_node["type"] == "I":# only consider I-nodes
                    outbounds.append(out_node) # add the node from another end
            elif edge["toID"] == node["nodeID"]:# inbound edge
                in_node = node_maps[edge['fromID']]
                if in_node["type"] == "I":
                    inbounds.append(in_node) # add the node from another end
        if len(inbounds) > 1:
            print("multiple inbound edges in nodeID:{} type:{}".format(node["nodeID"],node["type"]))
        if len(outbounds) > 1:
            print("multiple outbound edges in nodeID:{} type:{}".format(node["nodeID"],node["type"]))
        res.append({"head":inbounds,"relation":node,"tail":outbounds})
    return res

def get_new_data(data_path):
    with open(data_path,"r",encoding="utf-8") as f:
        data = f.read()
        data = json.loads(data)
    nodes = data["nodes"]
    edges = data["edges"]
    locutions = data["locutions"]
    # get nodes
    relation_nodes = {"RA":[],"CA":[],"MA":[]}
    tmp_node_map = {}
    for node in nodes:
        tmp_node_map.update({node["nodeID"]:node})
        if node["type"] == "RA":
            relation_nodes["RA"].append(node)
        elif node["type"] == "CA":
            relation_nodes["CA"].append(node)
        elif node["type"] == "MA":
            relation_nodes["MA"].append(node)

    # get inbound and outbound edges
    res_RA = get_relation_tuples(relation_nodes["RA"],edges,tmp_node_map)
    res_CA = get_relation_tuples(relation_nodes["CA"],edges,tmp_node_map)
    res_MA = get_relation_tuples(relation_nodes["MA"],edges,tmp_node_map)
    
    res_No = []
    def checknode(node,relation_nodes):
        final_nodes = relation_nodes["RA"] + relation_nodes["CA"] + relation_nodes["MA"]
        if not node["type"] == "I":
            return False
        for x in final_nodes :
            if node["nodeID"] == x["nodeID"]:
                return False
        return True
    
    for i in range(int(len(res_RA+res_MA+res_CA)*1)):
        while True:
            new_idx_1 = random.randint(0,len(nodes)-1)
            new_idx_2 = random.randint(0,len(nodes)-1)
            new_node1, new_node2 = nodes[new_idx_1],nodes[new_idx_2]
            if checknode(new_node1,relation_nodes) and checknode(new_node2,relation_nodes):
                res_No.append({"head":[new_node1],"relation":{"type":"None"},"tail":[new_node2]})
                break
    
    return res_RA , res_CA , res_MA, res_No

def process_phase_1(data):
    processed_data = []
    for triple in data:
        input_texts = ["HEAD:"]
        for i in range(len(triple["head"])):
            input_texts.append("{}.{}".format(i+1,triple["head"][i]["text"]))
        input_texts.append("  TAIL:")
        for i in range(len(triple["tail"])):
            input_texts.append("{}.{}".format(i+1,triple["tail"][i]["text"]))    
        input_texts.append("  HAVING RELATION:")
        input_texts.append("A. yes, B. no")
        input_text = ''.join(input_texts)
        if triple["relation"]["type"] == "None":
            processed_data.append({"text":input_text,"labels":"no"})
        else:
            processed_data.append({"text":input_text,"labels":"yes"})
    return processed_data

def process_phase_2(data):
    processed_data = []
    for triple in data:
        input_texts = ["HEAD:"]
        for i in range(len(triple["head"])):
            input_texts.append("{}.{}".format(i+1,triple["head"][i]["text"]))
        input_texts.append("  TAIL:")
        for i in range(len(triple["tail"])):
            input_texts.append("{}.{}".format(i+1,triple["tail"][i]["text"]))    
        input_texts.append("  RELATION:")
        input_texts.append("A. RA; B. CA; C. MA")
        input_text = ''.join(input_texts)
        processed_data.append({"text":input_text,"labels":triple["relation"]["type"]})
    return processed_data


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="./data/train/my_train")
    parser.add_argument("--data_path",type=str,default="./data/train/train_data/nodeset17918.json")
    parser.add_argument("--save_dir",type=str,default="./data/train/inode_relation_data_p1/")
    parser.add_argument("--eval_rate",type=float,default=0.05)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--neg_sample",type=int,default=5000)
    args = parser.parse_args()
    new_ra,new_ca,new_ma,new_no = [],[],[],[]
    if not args.data_dir:
        tmp_data = get_new_data(args.data_path)
        new_ra += tmp_data[0]
        new_ca += tmp_data[1]
        new_ma += tmp_data[2]
        new_no += tmp_data[3]
    else:
        datafiles = os.listdir(args.data_dir)
        for subdata in datafiles:
            tmp_data = get_new_data(os.path.join(args.data_dir,subdata))
            new_ra += tmp_data[0]
            new_ca += tmp_data[1]
            new_ma += tmp_data[2]
            new_no += tmp_data[3]
        
    print(f"RA NUM:{len(new_ra)}  CA NUM:{len(new_ca)}  MA NUM:{len(new_ma)}  NONE NUM:{len(new_no)}")
    
    random.seed(42)
    
    random.shuffle(new_ra)
    random.shuffle(new_ca)
    random.shuffle(new_ma)
    random.shuffle(new_no)
    ra_eval_num = int(len(new_ra)*0.05)
    ca_eval_num = int(len(new_ca)*0.05)
    ma_eval_num = int(len(new_ma)*0.05)
    no_eval_num = int(len(new_no)*0.05)
    
    print(f"RA EVAL NUM:{ra_eval_num }  CA EVAL NUM:{ca_eval_num }  MA EVAL NUM:{ma_eval_num }  NONE EVAL NUM:{no_eval_num }")
    
    
    eval_data = new_ra[:ra_eval_num] + new_ca[:ca_eval_num] + new_ma[:ma_eval_num] + new_no[:no_eval_num]
    train_data = new_ra[ra_eval_num:] + new_ca[ca_eval_num:] + new_ma[ma_eval_num:] + new_no[no_eval_num:]
    # eval_data = new_ra[:ra_eval_num] + new_ca[:ca_eval_num] + new_ma[:ma_eval_num]
    # train_data = new_ra[ra_eval_num:] + new_ca[ca_eval_num:] + new_ma[ma_eval_num:]
    
    random.shuffle(eval_data)
    random.shuffle(train_data)
    
    # count = 0
    # for tpl in new_data:
    #     if len(tpl["head"]) > 1:
    #         count += 1

    # print(f"num of multiple inbound nodes:{count}, total tuples num:{len(new_data)}")

    
    with open(os.path.join(args.save_dir,"train.json"),"w",encoding="utf-8") as saved_f:
        json.dump({"data":process_phase_1(train_data)},saved_f,ensure_ascii=False,indent=4)
    with open(os.path.join(args.save_dir,"eval.json"),"w",encoding="utf-8") as saved_f:
        json.dump({"data":process_phase_1(eval_data)},saved_f,ensure_ascii=False,indent=4)