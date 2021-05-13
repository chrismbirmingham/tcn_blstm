import os
import csv
import torch
import numpy
import pandas
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from threading import Thread


def multiply(outputs, gaze_mult):
    outputs1 =outputs[:,1] *gaze_mult
    outputs0 =outputs[:,0] /gaze_mult
    return numpy.array(list(zip(outputs0,outputs1)), dtype=object)


def score(targets, outputs):
    predictions = numpy.argmax(outputs,axis=1)
    outputs1 =outputs[:,1]
    outputs0 =outputs[:,0]
    targets = targets.flatten()
    acc = sum(targets==predictions)/len(targets)
    f1 = f1_score(targets, predictions)
    auroc = roc_auc_score(targets, outputs1)
    AP1 = average_precision_score(targets, outputs1)
    AP0 = average_precision_score(list(1-targets), outputs0)
    mAP = (AP1 + AP0)/2
    return acc, f1, auroc, mAP


def collect_data(data_path, model_type, feature_type, label_type, model_trainer):
    folders = os.listdir(data_path)

    all_labels, all_confidences, all_gazes = [], [], []

    for folder in folders:
        if not os.path.isdir(os.path.join(data_path, folder)):
            continue
        
        labels = numpy.load(os.path.join(data_path, folder, f"{folder}_ALL_LABELS_{label_type}.npy"))
        all_conf = pandas.read_csv(os.path.join(data_path, folder, f"{folder}_{model_type}_{feature_type}_{label_type}_conf{model_trainer}.csv"),usecols=["0Conf","1Conf"])

        # confidences = pandas.read_csv(os.path.join(data_path, folder, f"{folder}_{model_type}_{feature_type}_{label_type}_conf.csv"),usecols=["0Conf","1Conf"])
        gazes = pandas.read_csv(os.path.join(data_path, folder, f"{folder}_gaze_feat.csv"),usecols=["p1_ang", "p2_ang","p1_at", "p2_at"])

        labels = labels[:all_conf.shape[0]]

        gazes = gazes[8:all_conf.shape[0]+8]


        gazes = gazes.to_numpy(copy=True)

        # labels = labels.to_numpy(copy=True)
        confidences = all_conf.to_numpy(copy=True)

        assert len(labels)==len(gazes)==len(confidences), "all must be equal"

        all_labels.append(labels)
        all_confidences.append(confidences)
        all_gazes.append(gazes)


    all_confidences = numpy.concatenate(all_confidences)
    all_labels = numpy.concatenate(all_labels)
    all_gazes = numpy.concatenate(all_gazes)

    return all_confidences, all_labels, all_gazes

def add(outputs,gaze_mult, ratio=[2,1]):
    outputs1 =((outputs[:,1]*ratio[0]) + (gaze_mult * ratio[1]))/(ratio[0]+ratio[1])
    outputs0 =((outputs[:,0]*ratio[0]) + ((1-gaze_mult) * ratio[1]))/(ratio[0]+ratio[1])

    return numpy.array(list(zip(outputs0,outputs1)), dtype=object)


def check_adding(all_confidences, all_labels, all_gazes):
    lower_bound = 0
    upper_bound = 1
    m = (lower_bound-upper_bound)/75
    gazes_mult = (all_gazes[:,0] + all_gazes[:,1]) *.5  * (180/math.pi) * m + upper_bound
    
    all_new_conf = add(all_confidences,gazes_mult, ratio=[2,1])
    acc2, f12, auroc2, mAP2 = score(all_labels, all_new_conf)
    print(f"ANG  acc: {acc2-acc:.4f}, f1: {f12-f1:.4f}, auroc: {auroc2-auroc:.4f}, mAP: {mAP2-mAP:.4f}")

    gazes_mult = (all_gazes[:,2] + all_gazes[:,3]) *.5
    all_new_conf = add(all_confidences,gazes_mult, ratio=[2,1])
    acc2, f12, auroc2, mAP2 = score(all_labels, all_new_conf)
    print(f"AT acc: {acc2-acc:.4f}, f1: {f12-f1:.4f}, auroc: {auroc2-auroc:.4f}, mAP: {mAP2-mAP:.4f}")
    return


def main(data_path, model_type, feature_type, label_type, model_trainer=""):
    all_confidences, all_labels, all_gazes = collect_data(data_path, model_type, feature_type, label_type, model_trainer)
    all_confidences = math.e**all_confidences

    acc, f1, auroc, mAP = score(all_labels, all_confidences)
    print(f" acc: {acc:.4f}, f1: {f1:.4f}, auroc: {auroc:.4f}, mAP: {mAP:.4f}")

    # check_adding(all_confidences, all_labels, all_gazes)

    print("data collected")

    
    print("**************ANG*******************")
    ang_dict = {"lower":[0],"upper":[0],"acc":[acc],"f1":[f1],"auROC":[auroc],"mAP":[mAP]}
    # for lower_bound in [.5, .6, .7, .8, .9]:
    #     for upper_bound in [1, 1.1, 1.2, 1.3, 1.4, 1.5]:
    for i in range(0,5,1):
        lower_bound = i/10
        for j in range(5,15,2):
            upper_bound = j/10

            m = (lower_bound-upper_bound)/75
            gazes_mult = (all_gazes[:,0] + all_gazes[:,1]) *.5  * (180/math.pi) * m + upper_bound
            
            all_new_conf = multiply(all_confidences,gazes_mult)
            acc2, f12, auroc2, mAP2 = score(all_labels, all_new_conf)

            print(f"lower :{lower_bound}, upper: {upper_bound} acc: {acc2-acc:.4f}, f1: {f12-f1:.4f}, auroc: {auroc2-auroc:.4f}, mAP: {mAP2-mAP:.4f}")
            ang_dict['lower'].append(lower_bound)
            ang_dict['upper'].append(upper_bound) 
            ang_dict['acc'].append(acc2-acc)
            ang_dict['f1'].append(f12-f1)
            ang_dict['auROC'].append(auroc2-auroc)
            ang_dict['mAP'].append(mAP2-mAP)
    ang_df = pandas.DataFrame(ang_dict)
    ang_df.to_csv(f"metrics/{model_type}_{feature_type}_{label_type}_ang_metrics{model_trainer}.csv")


    print("**************AT*******************")
    at_dict = {"base":[0],"perc":[0],"acc":[acc],"f1":[f1],"auROC":[auroc],"mAP":[mAP]}
    
    for i in range(1,11,2):
        base = i/10
        for j in range(1,26,5):
            perc = j/100

            gazes_mult = base + all_gazes[:,1]*perc+all_gazes[:,2]*perc
            all_new_conf = multiply(all_confidences,gazes_mult)
            acc2, f12, auroc2, mAP2 = score(all_labels, all_new_conf)

            print(f"base :{base}, perc: {perc} : acc: {acc2-acc:.4f}, f1: {f12-f1:.4f}, auroc: {auroc2-auroc:.4f}, mAP: {mAP2-mAP:.4f}")
            at_dict['base'].append(base)
            at_dict['perc'].append(perc) 
            at_dict['acc'].append(acc2-acc)
            at_dict['f1'].append(f12-f1)
            at_dict['auROC'].append(auroc2-auroc)
            at_dict['mAP'].append(mAP2-mAP)
    at_df = pandas.DataFrame(at_dict)
    at_df.to_csv(f"metrics/{model_type}_{feature_type}_{label_type}_at_metrics{model_trainer}.csv")


if __name__ == '__main__':
    threadlist = []
    for m in ["BLSTM"]:#"TCN",
        for f in ["SYNCNET","PERFECTMATCH"]:
            for l in ["SPEECH"]:#, "TURN"
                print(m,f,l)
                threadlist.append(Thread(target=main, args=["data",m,f,l, "kalin"]))
                threadlist[-1].start()
    # for m in ["TCN","BLSTM"]:
    #     for f in ["SYNCNET","PERFECTMATCH"]:
    #         for l in ["SPEECH", "TURN"]:
    #             print(m,f,l)
    #             threadlist.append(Thread(target=main, args=["data",m,f,l, ""]))
    #             threadlist[-1].start()
