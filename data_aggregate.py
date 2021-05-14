import pandas as pd
import numpy as np
import os

def load_data(session, person, data_path="data", model_trainer=""):
    """ Load all the data we are interested in for a single person-session.

    Must ensure start frames match and lengths are the same.

    Starting frame should correlate to the 8th frame of the original recording.

    Returns:
        Labels 
            Turn-Labels     [:](already at 8+26: from preprocess)
            Speech-Labels   [:](already at 8: from preprocess)
        New Confidences
            Syncnet-TCN-Turn
            Syncnet-TCN-Speech
            Syncnet-BLSTM-Turn
            Syncnet-BLSTM-Speech
            Perfectmatch-TCN-Turn
            Perfectmatch-TCN-Speech
            Perfectmatch-BLSTM-Turn
            Perfectmatch-BLSTM-Speech
        Confidences
            Syncnet         [4:](0 correlates with 4, so 4+4=8)
            Perfectmatch    [4:](0 correlates with 4, so 4+4=8)
        Features
            Other-Gaze      [8:]
            Head-Angles     [8:]
    """
    folder=f"{session}{person[0]}"

    if not os.path.isdir(os.path.join(data_path, folder)):
        print(f"No folder for {folder}")
        return

    # Labels
    df_dict = {}
    for label_type in ["SPEECH","TURN"]:
        labels = np.load(os.path.join(data_path, folder, f"{folder}_ALL_LABELS_{label_type}.npy"))
        labels_df = pd.DataFrame(labels,columns=[f"{label_type}-LABEL"])
        df_dict[label_type]=labels_df.reset_index(drop=True)

    # New Confidences
    for m in ["TCN","BLSTM"]:
        for f in ["SYNCNET","PERFECTMATCH"]:
            for l in ["SPEECH", "TURN"]:
                if model_trainer=="kalin" and l == "TURN":
                    continue
                all_conf = pd.read_csv(os.path.join(data_path, folder, f"{folder}_{m}_{f}_{l}_conf{model_trainer}.csv"),usecols=["0Conf","1Conf"])
                all_conf.columns = [f"{m}-{f}-{l}-{c}" for c in all_conf.columns]
                df_dict[f"{m}-{f}-{l}-CONF"] = all_conf.reset_index(drop=True)

    # Confidences
    sync_path=f"/media/chris/M2/2-Processed_Data/syncnet_confidences/pyavi/{folder}/framewise_confidences.csv"
    sync_conf = pd.read_csv(sync_path,usecols=["Confidence"])
    sync_conf.columns = ["sConf"]
    df_dict["SYNCNET"] = sync_conf[4:].reset_index(drop=True)

    perf_path=f"/media/chris/M2/2-Processed_Data/perfectmatch_confidences/pyavi/{folder}/framewise_confidences.csv"
    perf_conf = pd.read_csv(perf_path,usecols=["Confidence"])
    perf_conf.columns = ["pConf"]
    df_dict["PERFECTMATCH"] = perf_conf[4:].reset_index(drop=True)
    
    # Features
    gazes = pd.read_csv(os.path.join(data_path, folder, f"{folder}_gaze_feat.csv"),usecols=["p1_ang", "p2_ang","p1_at", "p2_at"])
    df_dict["OTHER-GAZE"] = gazes[8:].reset_index(drop=True)

    opencv_path=f"/media/chris/M2/2-Processed_Data/Video-OpenFace-headpose/{session}/{person}.csv"
    poses = pd.read_csv(opencv_path,usecols=["pose_Rx","pose_Ry" ])
    df_dict["HEAD-ANGLEs"] = poses[8:].reset_index(drop=True)


    # Shorten dfs to same length
    m = min([df.shape[0] for k,df in df_dict.items()])
    for k,df in df_dict.items():
        df_dict[k] = df[:m]

    full_df = pd.concat([df for k,df in df_dict.items()],axis=1)
    full_df.to_feather(os.path.join(data_path, folder, f"analysis_df_{model_trainer}.feather"))

    return df_dict

    # confidences = pandas.read_csv(os.path.join(data_path, folder, f"{folder}_{model_type}_{feature_type}_{label_type}_conf.csv"),usecols=["0Conf","1Conf"])

    # labels = labels[:all_conf.shape[0]]

    # gazes = gazes[8:all_conf.shape[0]+8]

    # labels_30fps = pd.read_csv(os.path.join("data", folder, folder + '_VAD_MANUAL.csv'),usecols=["speech_activity"])
    # labels_25fps = labels_30fps[labels_30fps.index % 6 != 0].reset_index(drop=True)
    


    # labels = labels_25fps.values
for i in range(28):
    for person in ["left","right","center"]:
        _ = load_data(i, person, data_path="data", model_trainer="kalin")
