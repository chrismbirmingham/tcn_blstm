import pandas as pd
import numpy as np
import os

def load_chris_data(session, person, data_path="data", trained_on=""):
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
    for layers in ["1LAYER","2LAYER"]:
        for m in ["TCN","BLSTM"]:
            for f in ["SYNCNET","PERFECTMATCH"]:
                for l in ["SPEECH", "TURN"]:
                    if trained_on == "FOVA": trainer="kalin"
                    if trained_on == "RFSG": trainer="chris"
                    all_conf = pd.read_csv(os.path.join(data_path, folder, f"{folder}_CONF/{trainer}_{layers}_{l}_{m}_{f}.csv"),usecols=["0Conf","1Conf"])
                    all_conf.columns = [f"{layers}_{l}_{m}_{f}-{c}" for c in all_conf.columns]
                    df_dict[f"{layers}_{l}_{m}_{f}-CONF"] = all_conf.reset_index(drop=True)

    # Confidences
    sync_path=f"/home/chris/code/syncnet_python/data/syncnet_output/pyavi/{folder}/framewise_confidences.csv"
    sync_conf = pd.read_csv(sync_path,usecols=["Confidence"])
    sync_conf.columns = ["sConf"]
    df_dict["SYNCNET"] = sync_conf[8:].reset_index(drop=True)

    perf_path=f"/home/chris/code/syncnet_python/data/perfectmatch_output/pyavi/{folder}/framewise_confidences.csv"
    perf_conf = pd.read_csv(perf_path,usecols=["Confidence"])
    perf_conf.columns = ["pConf"]
    df_dict["PERFECTMATCH"] = perf_conf[8:].reset_index(drop=True)
    
    # Features
    gazes = pd.read_csv(os.path.join(data_path, folder, f"{folder}_gaze_feat.csv"),usecols=["p1_ang", "p2_ang","p1_at", "p2_at"])
    df_dict["OTHER-GAZE"] = gazes[8:].reset_index(drop=True)

    opencv_path=f"/media/chris/M2/2-Processed_Data/Video-OpenFace/{session}/{person}.csv"
    poses = pd.read_csv(opencv_path,usecols=["pose_Rx","pose_Ry" ])
    df_dict["HEAD-ANGLEs"] = poses[8:].reset_index(drop=True)


    # Shorten dfs to same length
    m = min([df.shape[0] for k,df in df_dict.items()])
    for k,df in df_dict.items():
        df_dict[k] = df[:m]

    full_df = pd.concat([df for k,df in df_dict.items()],axis=1)
    full_df.to_feather(os.path.join(data_path, folder, f"adf_trained_on_{trained_on}.feather"))

    return df_dict
   

def load_kalin_data(session, person, data_path="data", trained_on=""):
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
    feature_path="/home/chris/code/syncnet_python/scoring/kinect_pose"

    folder=f"{session}G3{person[0].capitalize()}"

    if not os.path.isdir(os.path.join(data_path, folder)):
        print(f"No folder for {folder}")
        return


    # Labels
    df_dict = {}
    # for label_type in ["SPEECH","TURN"]:
    #     labels = np.load(os.path.join(data_path, folder, f"{folder}_ALL_LABELS_{label_type}.npy"))
    #     labels_df = pd.DataFrame(labels,columns=[f"{label_type}-LABEL"])
    #     df_dict[label_type]=labels_df.reset_index(drop=True)
    # csv with a column for each speaker label with binary values for talking or not talking
    turns = pd.read_csv(f"{feature_path}/{session}G3_VAD.csv", usecols=[person])
    turns = turns[turns.index % 6 != 0].reset_index(drop=True)
    speech_df = turns[8:].reset_index(drop=True)
    speech_df.columns=["SPEECH-LABEL"]
    df_dict["SPEECH"] = speech_df
    turns_df = speech_df[8+26:].reset_index(drop=True)
    turns_df.columns=["TURN-LABEL"]
    df_dict["TURN"] = turns_df



    # New Confidences
    for layers in ["1LAYER","2LAYER"]:
        for m in ["TCN","BLSTM"]:
            for f in ["SYNCNET","PERFECTMATCH"]:
                for l in ["SPEECH", "TURN"]:

                    all_conf = pd.read_csv(os.path.join(data_path, folder, f"{folder}_RESULTS/CHRIS_{layers}_{f}_{m}_{l}.csv"),usecols=["0Conf","1Conf"])
                    all_conf.columns = [f"{layers}_{l}_{m}_{f}-{c}" for c in all_conf.columns]
                    df_dict[f"{layers}_{l}_{m}_{f}-CONF"] = all_conf.reset_index(drop=True)

    # Confidences
    # sync_path=f"/media/chris/M2/2-Processed_Data/syncnet_confidences/pyavi/{folder}/framewise_confidences.csv"
    # sync_conf = pd.read_csv(sync_path,usecols=["Confidence"])
    # csv with a single columns labeled "Confidence" and values from syncnet output
    sync_conf = pd.read_csv(f"{feature_path}/{folder}_SYNCNET.csv")
    sync_conf.columns = ["sConf"]
    df_dict["SYNCNET"] = sync_conf[8:].reset_index(drop=True)

    # perf_path=f"/media/chris/M2/2-Processed_Data/perfectmatch_confidences/pyavi/{folder}/framewise_confidences.csv"
    # perf_conf = pd.read_csv(perf_path,usecols=["Confidence"])
    perf_conf = pd.read_csv(f"{feature_path}/{folder}_PERFECTMATCH.csv")
    perf_conf.columns = ["pConf"]
    df_dict["PERFECTMATCH"] = perf_conf[8:].reset_index(drop=True)
    
    # Features
    # gazes = pd.read_csv(os.path.join(data_path, folder, f"{folder}_gaze_feat.csv"),usecols=["p1_ang", "p2_ang","p1_at", "p2_at"])
    # df_dict["OTHER-GAZE"] = gazes[8:].reset_index(drop=True)

    # opencv_path=f"/media/chris/M2/2-Processed_Data/Video-OpenFace-headpose/{session}/{person}.csv"
    # poses = pd.read_csv(opencv_path,usecols=["pose_Rx","pose_Ry" ])
    # df_dict["HEAD-ANGLEs"] = poses[8:].reset_index(drop=True)

    columns_of_interest = [f"{p}->{person}" for p in ["left", "right", "center"] if p != person]

    gaze_at_df = pd.read_csv(f"{feature_path}/{session}G3_KINECT_DISCRETE_EXTRALARGE.csv")
    gaze_at_df = gaze_at_df[gaze_at_df.index % 6 != 0].reset_index(drop=True)

    # csv with a column for each permutation of looker and subject with angle in radians
    # e.g. "left->right" | "left->center" | "right->left" | etc.
    gaze_ang_df = pd.read_csv(f"{feature_path}/{session}G3_KINECT_CONTINUOUS.csv")
    gaze_ang_df = gaze_ang_df[gaze_ang_df.index % 6 != 0].reset_index(drop=True)
   

    gaze_feat_df = gaze_ang_df[columns_of_interest]
    for p in ["left", "right", "center"]:
        if p != person:
            gaze_feat_df[f"{p}_at"] = (gaze_at_df[[p]]==person).astype(int)


    gaze_feat_df.columns = ["p1_ang", "p2_ang","p1_at", "p2_at"]
    df_dict["OTHER-GAZE"] = gaze_feat_df[8:].reset_index(drop=True)


    opencv_path=f"/home/chris/Downloads/openface_data/{folder}/{folder}_FACE.csv"
    poses = pd.read_csv(opencv_path, usecols=[" pose_Rx"," pose_Ry"])
    poses = poses[poses.index % 6 != 0].reset_index(drop=True)


    poses.columns = [c[1:] for c in poses.columns]
    df_dict["HEAD-ANGLEs"] = poses[8:].reset_index(drop=True)
    # opencv_path=f"/media/chris/M2/2-Processed_Data/Video-OpenFace-headpose/{session}/{person}.csv"
    # poses = pd.read_csv(opencv_path,usecols=["pose_Rx","pose_Ry" ])
    # df_dict["HEAD-ANGLEs"] = poses[8:].reset_index(drop=True)

    # Shorten dfs to same length
    m = min([df.shape[0] for k,df in df_dict.items()])
    for k,df in df_dict.items():
        print(k, df.shape)
        df_dict[k] = df[:m]

    full_df = pd.concat([df for k,df in df_dict.items()],axis=1)
    full_df.to_feather(os.path.join(data_path, folder, f"adf_trained_on_{trained_on}.feather"))

    return df_dict



# for i in range(1,28):
#     for person in ["left","right","center"]:
#         _ = load_chris_data(i, person, data_path="Data_RFSG", trained_on="FOVA")
#         _ = load_chris_data(i, person, data_path="Data_RFSG", trained_on="RFSG")



for i in range(1,16):
    for person in ["left","right","center"]:
        _ = load_kalin_data(i, person, data_path="Data_FOVA", trained_on="RFSG")
        # _ = load_kalin_data(i, person, data_path="Data_FOVA", trained_on="FOVA")
