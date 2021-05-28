import os
import csv
import torch
import numpy
import pandas
import random



def copy_dataset_format(drive_path):
    # Copy my dataset to the same format as Kalin's, with one person per folder
    sessions = os.listdir(os.path.join(drive_path,"Annotation-Turns"))
    for session in sessions:
        if not os.path.isdir(os.path.join(drive_path,"Annotation-Turns", session)):
            continue
        for person in ["left","right","center"]:
            personal_folder = session+person[0]
            data_folder = os.path.join("Data_RFSG",personal_folder)
            if not os.path.isdir(data_folder):
                os.makedirs(data_folder)
            
            turns = pandas.read_csv(os.path.join(drive_path,"Annotation-Turns", session, "turns.csv"))

            speaker_turns = turns[person].rename("speech_activity")

            speaker_turns.to_csv(os.path.join(data_folder,personal_folder+"_VAD_MANUAL.csv"))

            for net in ["syncnet", "perfectmatch"]:
                feat_dir = os.path.join("Data_RFSG",personal_folder,personal_folder+"_SYNCS","pywork",personal_folder)
                print(net, " features will be stored in: ", feat_dir)
                if not os.path.isdir(feat_dir):
                    os.makedirs(feat_dir)

                aud_path = f"/home/interactionlab/Downloads/{net}_output/pyfeat512/{personal_folder}/aud_feats.pt"
                vid_path = f"/home/interactionlab/Downloads/{net}_output/pyfeat512/{personal_folder}/vid_feats.pt"

                auds = torch.load(aud_path)
                vids = torch.load(vid_path)

                auds = auds.numpy()
                vids = vids.numpy()
                
                print("Video feature size", vids.shape)
                print("Audio feature size", auds.shape)
                
                numpy.save(os.path.join(feat_dir, f"feats_video_{net}.npy"),vids)
                numpy.save(os.path.join(feat_dir, f"feats_audio_{net}.npy"),auds)


def add_gaze_features(gaze_data_path, dataset="chris"):
    for session in range(1, 28):
        for person in ["left","right","center"]:
            folder_id = f"{session}{person[0]}"
            if folder_id not in os.listdir("Data_RFSG/"):
                print(f"Not including {folder_id}")
                continue

            # Gather gaze features into one df
            columns_of_interest = [f"{p}->{person}" for p in ["left", "right", "center"] if p != person]
            if dataset=="kalin":
                base_path = "/home/chris/code/modeling-pipeline/data/active_speaker/"
                gaze_at_df = pandas.read_csv(f"{base_path}/kinect_pose/{session}G3_KINECT_DISCRETE_EXTRALARGE.csv")

                # csv with a column for each permutation of looker and subject with angle in radians
                # e.g. "left->right" | "left->center" | "right->left" | etc.
                gaze_ang_df = pandas.read_csv(f"{base_path}/kinect_pose/{session}G3_KINECT_CONTINUOUS.csv")
            elif dataset=="chris":
                gaze_ang_df = pandas.read_csv(os.path.join(gaze_data_path, str(session), "pose_ang.csv"))
                gaze_at_df = pandas.read_csv(os.path.join(gaze_data_path, str(session), "pose_at_extralarge_cyl.csv"))

            gaze_feat_df = gaze_ang_df[columns_of_interest]
            for p in ["left", "right", "center"]:
                if p != person:
                    gaze_feat_df[f"{p}_at"] = (gaze_at_df[[p]]==person).astype(int)


            gaze_feat_df.columns = ["p1_ang", "p2_ang","p1_at", "p2_at"]

            # print(gaze_feat_df[:5])
            # input("continue?")
            gaze_feat_df_25fps = gaze_feat_df[gaze_feat_df.index % 6 != 0].reset_index(drop=True)
            shifted_gaze_feat_df_25fps = gaze_feat_df_25fps[4:]

            shifted_gaze_feat_df_25fps.to_csv(os.path.join("Data_RFSG",folder_id, f"{folder_id}_gaze_feat.csv"))



def gen_data(data_path):
    folders = os.listdir(data_path)
    for folder in folders:
        if not os.path.isdir(os.path.join(data_path, folder)):
            continue
        
        print('Preprocess: {}'.format(folder))
        
        for feature_type in ['video_syncnet', 'audio_syncnet', 'video_perfectmatch', 'audio_perfectmatch']:
            train_features_stack = []
            train_labels_stack = []
            val_features_stack = []
            val_labels_stack = []
            all_features_stack = []
            all_labels_stack = []

            features_tmp = numpy.load(os.path.join(data_path, folder, folder + '_SYNCS', 'pywork', folder, 'feats_' + feature_type + '.npy'))
            
            data_size = len(features_tmp)
            train_size = round(0.9 * data_size)
            val_size = data_size - train_size
            
            random.seed(0)
            split_idx = random.randint(0, data_size - val_size - 5)
            
            train_features_tmp1 = features_tmp[:split_idx]
            train_features_tmp2 = features_tmp[split_idx + val_size:]
            val_features_tmp = features_tmp[split_idx:split_idx + val_size]
            all_features_tmp = features_tmp
            
            train_features = []
            for i in range(5, len(train_features_tmp1) + 1):
                train_features.append(train_features_tmp1[i-5:i])
            for i in range(5, len(train_features_tmp2) + 1):
                train_features.append(train_features_tmp2[i-5:i])
            
            val_features = []
            for i in range(5, len(val_features_tmp) + 1):
                val_features.append(val_features_tmp[i-5:i])

            all_features = []
            for i in range(5, len(features_tmp) + 1):
                all_features.append(all_features_tmp[i-5:i])

            # Train and val should include all features but 4 at split_idx and 4 at split_idx+val_size
            assert len(train_features)+len(val_features)==len(all_features)-8, f"{len(train_features)}+{len(val_features)} doesn't = {len(all_features)-8}"
            
            train_features_stack.append(train_features)
            val_features_stack.append(val_features)
            all_features_stack.append(all_features)

            train_features_stack = sum(train_features_stack, [])
            val_features_stack = sum(val_features_stack, [])
            all_features_stack = sum(all_features_stack, [])
            
            labels_30fps = pandas.read_csv(os.path.join(data_path, folder, folder + '_VAD_MANUAL.csv'))
            labels_25fps = labels_30fps[labels_30fps.index % 6 != 0].reset_index(drop=True)
            
            labels_tmp = []
            for i, row in labels_25fps.iterrows():
                labels_tmp.append(int(row['speech_activity']))

            # SPEECH labels start at index 8
            train_labels_tmp1 = labels_tmp[:split_idx + 4]
            train_labels_tmp2 = labels_tmp[split_idx + val_size:len(all_features)+8]
            val_labels_tmp = labels_tmp[split_idx: split_idx + val_size + 4]
            all_labels_tmp = labels_tmp[:len(all_features)+8]
            
            train_labels_stack = [train_labels_tmp1[8:], train_labels_tmp2[8:]]
            train_labels_stack = sum(train_labels_stack, [])
            val_labels_stack = val_labels_tmp[8:]
            all_labels_stack = all_labels_tmp[8:]
            assert len(train_features_stack) == len(train_labels_stack), f"train feature len {len(train_features_stack)} must equal label len {len(train_labels_stack)}"
            assert len(val_features_stack) == len(val_labels_stack), f"val feature len {len(val_features_stack)} must equal label len {len(val_labels_stack)}"
            assert len(all_features_stack) == len(all_labels_stack), f"all feature len {len(all_features_stack)} must equal label len {len(all_labels_stack)}"

            
            numpy.save(os.path.join(data_path, folder, folder + '_' + feature_type.upper() + '_TRAIN_FEATURES_SPEECH.npy'), train_features_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_' + feature_type.upper() + '_VAL_FEATURES_SPEECH.npy'), val_features_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_' + feature_type.upper() + '_ALL_FEATURES_SPEECH.npy'), all_features_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_TRAIN_LABELS_SPEECH.npy'), train_labels_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_VAL_LABELS_SPEECH.npy'), val_labels_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_ALL_LABELS_SPEECH.npy'), all_labels_stack)

            print('\tSpeech Features: ({}) [{} | {}]\tLabels: [{} | {}]'.format(feature_type + '_speech', len(train_features_stack), len(val_features_stack), len(train_labels_stack), len(val_labels_stack)))
            
            # TURN labels start at index (8 + 26)
            train_labels_tmp1 = labels_tmp[:split_idx + 4 + 26]

            assert len(labels_tmp)>=len(all_features)+8, f"Not enough labels, have {len(labels_tmp)}, want {len(all_features)+8}"
            train_labels_tmp2 = labels_tmp[split_idx + val_size:len(all_features)+8]
            # Shorten train features because we don't have labels past the end of the video
            train_features_stack = train_features_stack[:-26]

            val_labels_tmp = labels_tmp[split_idx:split_idx + val_size + 4 + 26]

            assert len(labels_tmp)>=len(all_features)+8, f"Not enough labels, have {len(labels_tmp)}, want {len(all_features)+8}"
            all_labels_tmp = labels_tmp[:len(all_features)+8]
            # Shorten all features because we don't have labels past the end of the video
            all_features_stack = all_features_stack[:-26]


            train_labels_stack = [train_labels_tmp1[(8 + 26):], train_labels_tmp2[(8 + 26):]]
            val_labels_stack = val_labels_tmp[(8 + 26):]
            all_labels_stack = all_labels_tmp[(8 + 26):]
            train_labels_stack = sum(train_labels_stack, [])

            assert len(train_features_stack) == len(train_labels_stack), f"train feature len {len(train_features_stack)} must equal label len {len(train_labels_stack)}"
            assert len(val_features_stack) == len(val_labels_stack), f"val feature len {len(val_features_stack)} must equal label len {len(val_labels_stack)}"
            assert len(all_features_stack) == len(all_labels_stack), f"all feature len {len(all_features_stack)} must equal label len {len(all_labels_stack)}"

            numpy.save(os.path.join(data_path, folder, folder + '_' + feature_type.upper() + '_TRAIN_FEATURES_TURN.npy'), train_features_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_' + feature_type.upper() + '_VAL_FEATURES_TURN.npy'), val_features_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_' + feature_type.upper() + '_ALL_FEATURES_TURN.npy'), all_features_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_TRAIN_LABELS_TURN.npy'), train_labels_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_VAL_LABELS_TURN.npy'), val_labels_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_ALL_LABELS_TURN.npy'), all_labels_stack)


            print('\tTurn Features: ({}) [{} | {}]\tLabels: [{} | {}]'.format(feature_type + '_turn', len(train_features_stack), len(val_features_stack), len(train_labels_stack), len(val_labels_stack)))

def main():
    # copy_dataset_format("/media/interactionlab/M1/2- Processed Data/")
    add_gaze_features("/media/interactionlab/M1/2- Processed Data/Gaze-Data")
    
    gen_data('Data_RFSG')

if __name__ == '__main__':
    main()