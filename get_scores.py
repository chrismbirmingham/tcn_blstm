import json


root_path = "/home/interactionlab/chrisb/tcn_blstm/checkpoints"


if __name__ == '__main__':
    trainer = "chris"
    layers = 2
    for features in ["PERFECTMATCH","SYNCNET"]:# "PERFECTMATCH","SYNCNET"
        for model in ["BLSTM","TCN"]:# "BLSTM","TCN"
            for label in ["TURN", "SPEECH"]:# "TURN", "SPEECH"
                mAPs = []
                fpath = f"{root_path}/chris-l6o/2LAYER/{label}/{model}_{features}"
                for i in range(1,11):
                    full_path = f"{fpath}/{i}-fold_test_scores.json"
                    with open(full_path,'r') as s:
                        scores = json.load(s)
                        mAPs.append(scores["mAP"])
                mAP = sum(mAPs)/len(mAPs)
                # pmaps = 
                print(features,model,label,f"{mAP:.03f}")