import os
import csv
import numpy
import pandas
import torch


# File Paths
# /media/chris/M2/2-Processed_Data/syncnet_output/pyfeat512/1c/aud_feats.pt
# /media/chris/M2/2-Processed_Data/syncnet_output/pyfeat512/1c/vid_feats.pt
# /media/chris/M2/2-Processed_Data/Annotation-Turns/1/turns.csv

def copy_dataset_format(drive_path):
    # Copy my dataset to the same format as Kalin's, with one person per folder
    sessions = os.listdir(os.path.join(drive_path,"Annotation-Turns"))
    for session in sessions:
        if not os.path.isdir(os.path.join(drive_path,"Annotation-Turns", session)):
            continue
        if session in ["17", "21", "26"]:
            continue
        for person in ["left","right","center"]:
            personal_folder = session+person[0]
            data_folder = os.path.join("data",personal_folder)
            if not os.path.isdir(data_folder):
                os.makedirs(data_folder)
            
            turns = pandas.read_csv(os.path.join(drive_path,"Annotation-Turns", session, "turns.csv"))

            speaker_turns = turns[person].rename("speech_activity")

            speaker_turns.to_csv(os.path.join(data_folder,personal_folder+"_VAD_MANUAL.csv"))

            for net in ["syncnet", "perfectmatch"]:
                feat_dir = os.path.join("data",personal_folder,personal_folder+"_SYNCS","pywork",personal_folder)
                print(net, " features will be stored in: ", feat_dir)
                if not os.path.isdir(feat_dir):
                    os.makedirs(feat_dir)

                aud_path = f"/media/chris/M2/2-Processed_Data/{net}_output/pyfeat512/{personal_folder}/aud_feats.pt"
                vid_path = f"/media/chris/M2/2-Processed_Data/{net}_output/pyfeat512/{personal_folder}/vid_feats.pt"

                auds = torch.load(aud_path)
                vids = torch.load(vid_path)

                auds = auds.numpy()
                vids = vids.numpy()
                
                print("Video feature size", vids.shape)
                print("Audio feature size", auds.shape)
                
                numpy.save(os.path.join(feat_dir, f"feats_video_{net}.npy"),vids)
                numpy.save(os.path.join(feat_dir, f"feats_audio_{net}.npy"),auds)






def gen_data(data_path):
    folders = os.listdir(data_path)
    for folder in folders:
        if not os.path.isdir(os.path.join(data_path, folder)):
            continue
        
        print('Preprocess {}'.format(folder))
        
        for feature_type in ['video_syncnet', 'audio_syncnet', 'video_perfectmatch', 'audio_perfectmatch']:
            features_stack = []
            labels_stack = []

            labels_30fps = pandas.read_csv(os.path.join(data_path, folder, folder + '_VAD_MANUAL.csv'))
            labels_25fps = labels_30fps[labels_30fps.index % 6 != 0].reset_index(drop=True)
            
            labels_tmp = []
            for i, row in labels_25fps.iterrows():
                labels_tmp.append(int(row['speech_activity']))

            features_tmp = numpy.load(os.path.join(data_path, folder, folder + '_SYNCS', 'pywork', folder, 'feats_' + feature_type + '.npy'))
            
            features = []
            for i in range(5, len(features_tmp) + 1):
                features.append(features_tmp[i-5:i])
            
            labels = labels_tmp[4:len(features_tmp)]

            features_stack.append(features)
            labels_stack.append(labels)
            
            if not len(features) == len(labels):
                print(f'33[31m\n\t\Size missmatch {folder}\tDatapoints {len(features)}/{len(labels)}')
                break
        
            features_stack = sum(features_stack, [])
            labels_stack = sum(labels_stack, [])
        
            numpy.save(os.path.join(data_path, folder, folder + '_' + feature_type.upper() + '_FEATURES.npy'), features_stack)
            numpy.save(os.path.join(data_path, folder, folder + '_LABELS.npy'), labels_stack)
            
            print('\tDatapoints ({}) [{}]'.format(feature_type, len(features)))

def main():
    # copy_dataset_format("/media/chris/M2/2-Processed_Data/")
    gen_data('data')

if __name__ == '__main__':
    main()