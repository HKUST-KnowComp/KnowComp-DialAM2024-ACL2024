import json
import itertools
import os
from sklearn.metrics import precision_recall_fscore_support

def main(file):                        
    file1 = open(f'./eval/Evaluation_Data/{file}', 'r') # golden data
    truth = json.load(file1)["AIF"]
    file2 = open(f"./test_{file}","r") # my predicted data
    preds = json.load(file2)

    proposition_dict = {}
    proposition_list = []
    true_inference_list = []
    true_conflict_list = []
    true_rephrase_list = []
    pred_inference_list = []
    pred_conflict_list = []
    pred_rephrase_list = []

    # Get the list of proposition nodes
    for node in truth['nodes']:
        if node['type'] == "I":
            proposition_list.append(node['nodeID'])
            proposition_dict[node['nodeID']] = node['text']

    # Check truth relations
    for node in truth['nodes']:
        if node['type'] == "RA":
            inference_id = node['nodeID']

            for edge in truth['edges']:
                if edge['fromID'] == inference_id:
                    conclusion_id = edge['toID']
                    for edge in truth['edges']:
                        if edge['toID'] == inference_id:
                            premise_id = edge['fromID']
                            if premise_id in proposition_list:
                                true_inference_list.append([premise_id, conclusion_id, 0])
                    break

        elif node['type'] == "CA":
            conflict_id = node['nodeID']

            for edge in truth['edges']:
                if edge['fromID'] == conflict_id:
                    conf_to = edge['toID']
                    for edge in truth['edges']:
                        if edge['toID'] == conflict_id:
                            conf_from = edge['fromID']
                            if conf_from in proposition_list:
                                true_conflict_list.append([conf_from, conf_to, 1])
                    break

        elif node['type'] == "MA":
            rephrase_id = node['nodeID']

            for edge in truth['edges']:
                if edge['fromID'] == rephrase_id:
                    reph_to = edge['toID']
                    for edge in truth['edges']:
                        if edge['toID'] == rephrase_id:
                            reph_from = edge['fromID']
                            if reph_from in proposition_list:
                                true_rephrase_list.append([reph_from, reph_to, 2])
                    break

    # Check predicted relation
    for node in preds['nodes']:
        if node['type'] == "RA":
            inference_id = node['nodeID']

            for edge in preds['edges']:
                if edge['fromID'] == inference_id:
                    conclusion_id = edge['toID']
                    for edge in preds['edges']:
                        if edge['toID'] == inference_id:
                            premise_id = edge['fromID']
                            if premise_id in proposition_list:
                                pred_inference_list.append([premise_id, conclusion_id, 0])
                    break

        elif node['type'] == "CA":
            conflict_id = node['nodeID']

            for edge in preds['edges']:
                if edge['fromID'] == conflict_id:
                    conf_to = edge['toID']
                    for edge in preds['edges']:
                        if edge['toID'] == conflict_id:
                            conf_from = edge['fromID']
                            if conf_from in proposition_list:
                                pred_conflict_list.append([conf_from, conf_to, 1])
                    break

        elif node['type'] == "MA":
            rephrase_id = node['nodeID']

            for edge in preds['edges']:
                if edge['fromID'] == rephrase_id:
                    reph_to = edge['toID']
                    for edge in preds['edges']:
                        if edge['toID'] == rephrase_id:
                            reph_from = edge['fromID']
                            if reph_from in proposition_list:
                                pred_rephrase_list.append([reph_from, reph_to, 2])
                    break

    # ID to text

    p_c = itertools.permutations(proposition_list, 2)
    proposition_combinations = []
    for p in p_c:
        proposition_combinations.append([p[0], p[1]])

    y_true = []
    y_pred = []
    for comb in proposition_combinations:
        added_true = False
        added_pred = False

        # Prepare Y true
        for inference in true_inference_list:
            if inference[0] == comb[0] and inference[1] == comb[1]:
                y_true.append(0)
                added_true = True
                break
        if not added_true:
            for conflict in true_conflict_list:
                if conflict[0] == comb[0] and conflict[1] == comb[1]:
                    y_true.append(1)
                    added_true = True
                    break
        if not added_true:
            for rephrase in true_rephrase_list:
                if rephrase[0] == comb[0] and rephrase[1] == comb[1]:
                    y_true.append(2)
                    added_true = True
                    break
        if not added_true:
            y_true.append(3)

        # Prepare Y pred
        for inference in pred_inference_list:
            if inference[0] == comb[0] and inference[1] == comb[1]:
                y_pred.append(0)
                added_pred = True
                break
        if not added_pred:
            for conflict in pred_conflict_list:
                if conflict[0] == comb[0] and conflict[1] == comb[1]:
                    y_pred.append(1)
                    added_pred = True
                    break
        if not added_pred:
            for rephrase in pred_rephrase_list:
                if rephrase[0] == comb[0] and rephrase[1] == comb[1]:
                    y_pred.append(2)
                    added_pred = True
                    break
        if not added_pred:
            y_pred.append(3)

    focused_true = []
    focused_pred = []
    for i in range(len(y_true)):
        if y_true[i] != 3:
            focused_true.append(y_true[i])
            focused_pred.append(y_pred[i])

    return {"General":precision_recall_fscore_support(y_true, y_pred, average='macro',zero_division=0),"Focused":precision_recall_fscore_support(focused_true, focused_pred, average='macro',zero_division=0)}

if __name__=="__main__":
    scores = {"g_precision":0,"g_recall":0,"g_fscore":0,"f_precision":0,"f_recall":0,"f_fscore":0,"cnt":0}
    for file in os.listdir("./eval/Evaluation_Data"):
    # for file in ["nodeset18321.json"]:
        res = main(file)
        scores["g_precision"] += res["General"][0]
        scores["g_recall"] += res["General"][1]
        scores["g_fscore"] += res["General"][2]
        scores["f_precision"] += res["Focused"][0]
        scores["f_recall"] += res["Focused"][1]
        scores["f_fscore"] += res["Focused"][2]
        scores["cnt"] += 1
    print("General precision, recall, fscore: {},{},{}".format(scores["g_precision"]/scores["cnt"],scores["g_recall"]/scores["cnt"],scores["g_fscore"]/scores["cnt"]))
    print("Focused precision, recall, fscore: {},{},{}".format(scores["f_precision"]/scores["cnt"],scores["f_recall"]/scores["cnt"],scores["f_fscore"]/scores["cnt"]))