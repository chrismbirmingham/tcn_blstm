import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import os
import math
import json
from threading import Thread
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed

def score(targets, outputs):
    predictions = np.argmax(outputs,axis=1)
    outputs1 =outputs[:,1]
    outputs0 =outputs[:,0]
    targets = targets.flatten()
    acc = sum(targets==predictions)/len(targets)
    f1 = f1_score(targets, predictions)
    try:
        auroc = roc_auc_score(targets, outputs1)
    except ValueError as e:
        print(e, f"Class balance is: {sum(targets)/len(targets)}")
        auroc = None
    AP1 = average_precision_score(targets, outputs1)
    AP0 = average_precision_score(list(1-targets), outputs0)
    mAP = (AP1 + AP0)/2
    return acc, f1, auroc, mAP

def get_feathered_data(data_path="data",trained_on="FOVA"):
    df_list = []
    folders = os.listdir(data_path)
    for folder in folders:
        if not os.path.isdir(os.path.join(data_path, folder)):
            print("No folder here:",os.path.join(data_path, folder))
            continue
        if os.path.exists(f"{data_path}/{folder}/adf_trained_on_{trained_on}.feather"):
            df_list.append(pd.read_feather(os.path.join(data_path, folder, f"adf_trained_on_{trained_on}.feather")))
    full_df = pd.concat(df_list)


    # Fixing the df
    # Fixing multipliers to range
    lower_bound = 0
    upper_bound = 1
    m = (lower_bound-upper_bound)/75
    full_df["ang_mult"] = (full_df["p1_ang"].values + full_df["p2_ang"].values)*.5  * ((180/math.pi) * m) + upper_bound
    full_df["at_mult"] = (full_df["p1_at"].values + full_df["p2_at"].values)*.5

    # Fixing to [0,1] confidence scores
    for l in ["SPEECH", "TURN"]:
        for m in ["TCN","BLSTM"]:
            for f in ["SYNCNET","PERFECTMATCH"]:
                for n in ["1LAYER","2LAYER"]:
                    full_df[f'{n}_{l}_{m}_{f}-1Conf'] = math.e**full_df[f'{n}_{l}_{m}_{f}-1Conf'].values
                    full_df[f'{n}_{l}_{m}_{f}-0Conf'] = math.e**full_df[f'{n}_{l}_{m}_{f}-0Conf'].values
                    # full_df.drop([f'{n}_{l}_{m}_{f}-0Conf', f'{n}_{l}_{m}_{f}-1Conf'], axis=1, inplace=True)

    # Fixing sync and perf conf to [0,1]
    for m in ["sConf","pConf"]:
        full_df[m] = (full_df[m]/20)+.5
        full_df[m][full_df[m]<0]=0
        full_df[m][full_df[m]>1]=1


    # print(full_df.shape)
    # print(full_df.columns)
    return full_df

def score_df(df, label="SPEECH"):
    scores = {}
    pos_sum  = int(df[f"{label}-LABEL"].sum())
    scores["labels"] = {
        "total":int(df.shape[0]),
        "positive":pos_sum,
        "negative":int(df.shape[0]-pos_sum)
    }
    
    # Score original models
    for m in ["sConf","pConf"]:
        outputs1 = df[m].values
        outputs0 = 1-df[m].values
        outputs = np.array(list(zip(outputs0,outputs1)), dtype=object)

        acc, f1, auroc, mAP = score(df[f"{label}-LABEL"].values, outputs)
        scores[m] = {"acc":acc,"f1":f1,"auROC":auroc,"mAP":mAP}
        # print(m,{"acc":acc,"f1":f1,"auROC":auroc,"mAP":mAP})
        
    # # Score Gaze Features   
    # outputs1 = df["ang_mult"].values
    # outputs0 = 1-outputs1
    # outputs = np.array(list(zip(outputs0,outputs1)), dtype=object)

    # acc, f1, auroc, mAP = score(df[f"{label}-LABEL"].values, outputs)
    # scores[f'ang'] = {"acc":acc,"f1":f1,"auROC":auroc,"mAP":mAP}
    
    # outputs1 = df["at_mult"].values
    # outputs0 = 1-outputs1
    # outputs = np.array(list(zip(outputs0,outputs1)), dtype=object)

    # acc, f1, auroc, mAP = score(df[f"{label}-LABEL"].values, outputs)
    # scores[f'at'] = {"acc":acc,"f1":f1,"auROC":auroc,"mAP":mAP}
    
    # Score newly trained models
    for m in ["TCN","BLSTM"]:
        for f in ["SYNCNET","PERFECTMATCH"]:
            for n in ["1LAYER","2LAYER"]:
                # outputs1 = df[f'{n}_{label}_{m}_{f}-1Conf'].values
                # outputs0 = df[f'{n}_{label}_{m}_{f}-0Conf'].values
                # # outputs0 = 1-outputs1
                # outputs = np.array(list(zip(outputs0,outputs1)), dtype=object)

                acc, f1, auroc, mAP = score(df[f"{label}-LABEL"].values,  df[[f'{n}_{label}_{m}_{f}-0Conf', f'{n}_{label}_{m}_{f}-1Conf']].values)
                scores[f'{n}_{m}_{f}'] = {"acc":acc,"f1":f1,"auROC":auroc,"mAP":mAP}
    df = pd.DataFrame(scores)
    
    return scores, df

@timeit
def calc_perf_by_window(df, windows, split_col_name, plot=False, label="SPEECH"):
    results = []
    ticks = []
    

    for rmin, rmax in windows:
        #rmin = min(df[split_col_name]) + window * i
        #rmax = min(df[split_col_name]) + window * (i+1)
        sub_df = df[(df[split_col_name]>=rmin*math.pi/180) & (df[split_col_name]<rmax*math.pi/180)]

        tick = f"{rmin:.0f}:{rmax:.0f}"

        # print(f"Range is {rmin:.0f}, {rmax:.0f} and support is {len(sub_df)}")
        s, df_scores = score_df(sub_df, label=label)
        results.append(s)
        ticks.append(tick)
    return results, ticks

def avg_gaze(full_df, gaze_type="ang", weights=[1,1]):
    df = full_df.copy(deep=True)
    for c in df.columns:
        if c not in ["pose_Rx","pose_Ry", f"SPEECH-LABEL","TURN-LABEL","p1_ang","p2_ang","p1_at","p2_at","ang_mult","at_mult"]:
            df[c] = (df[c].values * weights[0] + df[f"{gaze_type}_mult"] * weights[1])/ sum(weights)
    return df

def mult_gaze(full_df, gaze_type="ang", weights=[1,1]):
    df = full_df.copy(deep=True)
    for c in df.columns:
        if c not in ["pose_Rx","pose_Ry", f"SPEECH-LABEL","TURN-LABEL","p1_ang","p2_ang","p1_at","p2_at","ang_mult","at_mult"]:
            df[c] = df[c].values * (df[f"{gaze_type}_mult"] + .5)
    return df

def get_multiplied_conf(full_df):
    dfs = {}

    for gaze_type in ["ang","at"]:
        dfs[gaze_type] = {}
        dfs[gaze_type]["avg"] = avg_gaze(full_df, gaze_type=gaze_type, weights=[1,5])
        dfs[gaze_type]["mult"] = mult_gaze(full_df, gaze_type=gaze_type)
    return dfs


def cal_perf_by_window_wrapper(results, num_buckets, gaze_type, mult_type, df, buckets, label):
    print(num_buckets, gaze_type, mult_type,label,"starting")
    results[num_buckets][label][gaze_type][mult_type], ticks = calc_perf_by_window(df, buckets, "pose_Ry", label=label)
    results[num_buckets]["xticks"]=ticks
    print(num_buckets, gaze_type, mult_type,label,"Finished")
    return

def get_results_on_window(dfs,full_df,num_buckets,results):
        print(f"Bucketing into {num_buckets} buckets")
        bucket_size = 120/num_buckets
        buckets = [(-60+i*bucket_size, -60+(i+1)*bucket_size) for i in range(num_buckets)]
        results[num_buckets]={}
        subthreads = []
        for label in ["SPEECH","TURN"]:
            results[num_buckets][label]={}
 
            for gaze_type, mult_type_dfs in dfs.items():
                results[num_buckets][label][gaze_type]={}

                for mult_type, df in mult_type_dfs.items():
                    subthreads.append(Thread(target=cal_perf_by_window_wrapper, args=(results, num_buckets, gaze_type, mult_type, df, buckets, label)))
                    subthreads[-1].start()  
                    # results[num_buckets][label][gaze_type][mult_type], ticks = calc_perf_by_window(df, buckets, "pose_Ry", label=label)
                    # results[num_buckets]["xticks"]=ticks
            for s in subthreads:
                s.join()
            print(num_buckets, label, "starting original")
            results[num_buckets][label]["original"],_ = calc_perf_by_window(full_df, buckets, "pose_Ry", label=label)
            print(num_buckets, label, "finishing original")


def get_all_results(dfs, full_df):
    results = {}
    bucket_sizes = [3,4,5,6,8]
    threads = [None] * len(bucket_sizes)

    for i, num_buckets in enumerate(bucket_sizes):
        threads[i] = Thread(target=get_results_on_window, args=(dfs, full_df, num_buckets, results))
        threads[i].start()
    for t in threads:
        t.join()
    print(results)
    return results



def score_and_save(trained_on,tested_on):
    data_path = f"Data_{tested_on}"
    full_df = get_feathered_data(data_path=data_path, trained_on=trained_on)

    dfs = get_multiplied_conf(full_df)
    results = get_all_results(dfs, full_df)

    with open(f'scored_results/trained_on_{trained_on}_tested_on_{tested_on}_results.json', 'w') as f:

        json_obj = json.dumps(results)
        f.write(json_obj)
        f.close()


if __name__ == '__main__':
    score_and_save("RFSG","FOVA")
    # score_and_save("FOVA","RFSG")
    # score_and_save("RFSG","RFSG")