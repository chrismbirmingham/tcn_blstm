import os
import csv
import torch
import numpy
import pandas
import random
import math
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


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


def collect_data(data_path, model="TCN", base_model="PERFECTMATCH"):
    folders = os.listdir(data_path)

    all_labels, all_confidences, all_gazes = [], [], []

    for folder in folders:
        if not os.path.isdir(os.path.join(data_path, folder)):
            continue
        
        # print('Score {}'.format(folder))

        labels_30fps = pandas.read_csv(os.path.join(data_path, folder, folder + '_VAD_MANUAL.csv'),usecols=["speech_activity"])
        labels = labels_30fps[labels_30fps.index % 6 != 0].reset_index(drop=True)
        labels = labels[4:]

        confidences = pandas.read_csv(os.path.join(data_path, folder, f"{folder}_{model}_{base_model}_conf_kalin.csv"),usecols=["0Conf","1Conf"])
        gazes = pandas.read_csv(os.path.join(data_path, folder, f"{folder}_gaze_feat.csv"),usecols=["p1_ang", "p2_ang","p1_at", "p2_at"])

        labels = labels[:confidences.shape[0]]
        gazes = gazes[:confidences.shape[0]]


        gazes = gazes.to_numpy(copy=True)

        labels = labels.to_numpy(copy=True)
        confidences = confidences.to_numpy(copy=True)

        all_labels.append(labels)
        all_confidences.append(confidences)
        all_gazes.append(gazes)

        # try:
        #     acc, f1, auroc, mAP = score(labels, confidences)
        #     print("Continue?"+f" acc: {acc:.4f}, f1: {f1:.3f}, auroc: {auroc:.3f}, mAP: {mAP:.3f},")
        #     acc, f1, auroc, mAP = score(labels, new_conf)
        #     print("Continue?"+f" acc: {acc:.3f}, f1: {f1:.3f}, auroc: {auroc:.3f}, mAP: {mAP:.3f},")
        # except Exception as e:
        #     print(e)


    all_confidences = numpy.concatenate(all_confidences)
    all_labels = numpy.concatenate(all_labels)
    all_gazes = numpy.concatenate(all_gazes)

    return all_confidences, all_labels, all_gazes


def score_model(data_path, model="TCN", base_model="PERFECTMATCH"):
    all_confidences, all_labels, all_gazes = collect_data(data_path, model, base_model)
    print("FINAL")

    acc, f1, auroc, mAP = score(all_labels, all_confidences)
    print(f" acc: {acc:.4f}, f1: {f1:.4f}, auroc: {auroc:.4f}, mAP: {mAP:.4f}")
    
    print("**************ANG*******************")
    for lower_bound in [.5, .6, .7, .8, .9]:
        for upper_bound in [1, 1.1, 1.2, 1.3, 1.4, 1.5]:
            print(f"\n{lower_bound}, {upper_bound}")

            m = (lower_bound-upper_bound)/75
            gazes_mult = (all_gazes[:,2] + all_gazes[:,3]) *.5  * (180/math.pi) * m + upper_bound
            
            all_new_conf = multiply(all_confidences,gazes_mult)
            acc2, f12, auroc2, mAP2 = score(all_labels, all_new_conf)
            # print(f" acc: {acc2:.4f}, f1: {f12:.4f}, auroc: {auroc2:.4f}, mAP: {mAP2:.4f}")

            print(f"Improvement: acc: {acc2-acc:.4f}, f1: {f12-f1:.4f}, auroc: {auroc2-auroc:.4f}, mAP: {mAP2-mAP:.4f}")

    print("**************AT*******************")
    
    for base in [.7, .8, .9, 1, 1.1]:
        for m in [.05, .15, .25, .5, .75]:
            print(f"\n{base}, {m}")
            gazes_mult = base + all_gazes[:,0]*m+all_gazes[:,1]*m
            all_new_conf = multiply(all_confidences,gazes_mult)
            acc2, f12, auroc2, mAP2 = score(all_labels, all_new_conf)
            # print(f" acc: {acc2:.4f}, f1: {f12:.4f}, auroc: {auroc2:.4f}, mAP: {mAP2:.4f}")

            print(f"Improvement: acc: {acc2-acc:.4f}, f1: {f12-f1:.4f}, auroc: {auroc2-auroc:.4f}, mAP: {mAP2-mAP:.4f}")
    


    input("continue?")


if __name__ == '__main__':
    for m in ["TCN","BLSTM"]:
        for f in ["PERFECTMATCH"]:#"SYNCNET",
            print(m,f) 
            score_model('data',m,f)