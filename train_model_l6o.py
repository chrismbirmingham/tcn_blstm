import os, random
import torch
from tqdm import tqdm
import numpy
import pandas
import matplotlib.pyplot as plt
import json

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from tcn import TemporalConvNet

def list_contains(List1, List2): 
  
    set1 = set(List1) 
    set2 = set(List2) 
    if set1.intersection(set2): 
        return True 
    else: 
        return False

def custom_l6o_iterator(lengths):
    print(lengths)
    print(len(lengths))
    r = []
    for f in range(11):
        test_p = list(range(f*6,(f+1)*6))
        train_p = list(range(len(lengths)))
        for p in test_p:
            train_p.remove(p)
        train_idx, val_idx, test_idx = [], [], []
        current_final_idx = 0
        for i, l in enumerate(lengths):
            idxs = [n +current_final_idx for n in range(l)]
            if i in train_p:
                split_ind = random.randint(0,len(idxs))
                split_ind2 = split_ind + int(len(idxs)/10)
                train_idx += idxs[:split_ind]+idxs[split_ind2:]
                val_idx +=idxs[split_ind:split_ind2]

            elif i in test_p:
                test_idx += idxs
            current_final_idx += l
        r.append((train_idx, val_idx, test_idx))
    return r

class Model_BLSTM(torch.nn.Module):
    def __init__(self, num_layers):
        super(Model_BLSTM, self).__init__()
        
        self.nhid = 128
        
        self.lstm_v = torch.nn.LSTM(512, self.nhid, num_layers=num_layers, bidirectional=True)
        self.lstm_a = torch.nn.LSTM(512, self.nhid, num_layers=num_layers, bidirectional=True)
        self.fc = torch.nn.Linear(2*self.nhid*2*5, 2)

    def forward(self, x_v, x_a):
        x_v = self.lstm_v(x_v)[0]
        x_a = self.lstm_a(x_a)[0]
        x_v = x_v.view(-1, self.nhid*2*5)
        x_a = x_a.view(-1, self.nhid*2*5)
        
        x = self.fc(torch.cat((x_v, x_a), axis=1))
        
        return torch.nn.functional.log_softmax(x, dim=1)

class Model_TCN(torch.nn.Module):
    def __init__(self, num_layers):
        super(Model_TCN, self).__init__()
        
        self.nhid = 128
        
        self.tcn_v = TemporalConvNet(512, [self.nhid for i in range(num_layers)], 3, dropout=0.3)
        self.tcn_a = TemporalConvNet(512, [self.nhid for i in range(num_layers)], 3, dropout=0.3)
        self.fc = torch.nn.Linear(2*self.nhid, 2)

    def forward(self, x_v, x_a):
        x_v = self.tcn_v(x_v)
        x_a = self.tcn_a(x_a)
        x_v = x_v[:, :, -1]
        x_a = x_a[:, :, -1]
        
        x = self.fc(torch.cat((x_v, x_a), axis=1))
        
        return torch.nn.functional.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, criterion, log_interval, epoch, model_type="TCN"):
    model.train()

    f1s, aurocs, mAPs = [], [], []
    outputs0, outputs1, predictions, targets = [], [], [], []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data_v, data_a = data[:, :, :, 0], data[:, :, :, 1]
        
        data_v, data_a, target = torch.FloatTensor(data_v).to(device), torch.FloatTensor(data_a).to(device), torch.LongTensor(target).to(device)
        data_v, data_a, target = Variable(data_v), Variable(data_a), Variable(target)
        
        if model_type == "TCN":
            data_v = data_v.permute(0, 2, 1)
            data_a = data_a.permute(0, 2, 1)
        
        output = model(data_v, data_a)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        pred = pred.detach().cpu()
        predictions += pred.tolist()

        target = target.cpu().view_as(pred)
        targets += target.tolist()

        output1 = output.detach().cpu()[:,1]
        outputs1 += output1.tolist()

        output0 = output.detach().cpu()[:,0]
        outputs0 += output0.tolist()

        if batch_idx > 0 and batch_idx % log_interval == 0:
            f1 = f1_score(targets, predictions)
            auroc = roc_auc_score(targets, outputs1)
            AP1 = average_precision_score(targets, outputs1)
            AP0 = average_precision_score(list(1-numpy.array(targets)), outputs0)
            mAP = (AP1 + AP0)/2
            f1s.append(f1)
            aurocs.append(auroc)
            mAPs.append(mAP)
            outputs0, outputs1, predictions, targets = [], [], [], []
    return f1s, aurocs, mAPs

def validate(model, device, val_loader, criterion, epoch,  model_type="TCN"):
    model.eval()
    
    loss = 0
    correct = 0
    outputs0, outputs1, predictions, targets = [], [], [], []
    with torch.no_grad():
        for data, target in val_loader:
            data_v, data_a = data[:, :, :, 0], data[:, :, :, 1]
            
            data_v, data_a, target = torch.FloatTensor(data_v).to(device), torch.FloatTensor(data_a).to(device), torch.LongTensor(target).to(device)
            data_v, data_a, target = Variable(data_v), Variable(data_a), Variable(target)
            
            if model_type == "TCN":
                data_v = data_v.permute(0, 2, 1)
                data_a = data_a.permute(0, 2, 1)
        
            output = model(data_v, data_a)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            
            correct += pred.eq(target.view_as(pred)).sum().item()

            pred = pred.detach().cpu()
            predictions += pred.tolist()

            target = target.cpu().view_as(pred)
            targets += target.tolist()

            output1 = output.detach().cpu()[:,1]
            outputs1 += output1.tolist()

            output0 = output.detach().cpu()[:,0]
            outputs0 += output0.tolist()
    
    loss /= len(val_loader.dataset)
    acc = correct / len(val_loader.dataset)
    f1 = f1_score(targets, predictions)
    auroc = roc_auc_score(targets, outputs1)
    AP1 = average_precision_score(targets, outputs1)
    AP0 = average_precision_score(list(1-numpy.array(targets)), outputs0)
    mAP = (AP1 + AP0)/2

    print('\tLoss {:.5f}\tAccuracy {:.3f}\tF1: {:.3f}\tauROC: {:.3f}\tmAP: {:.3f}'.format(
        loss, acc, f1, auroc, mAP))
        
    return acc, f1, auroc, mAP

def calculate_class_weights(dataset, class_num):
    labels = []
    for data, target in dataset:
    	labels.append(target.item())
    
    weights = []
    for c in range(class_num):
        weights.append(len(labels) / labels.count(c))

    weights = [float(i) / sum(weights) for i in weights]

    return weights

def main(model_type, feature_type, label_type, num_layers, trainer="chris"):
    directory = f"checkpoints/{trainer}-l6o/{num_layers}LAYER/{label_type}/{model_type}_{feature_type}"

    if not os.path.exists(directory):
        os.makedirs(directory)
    patience=5
    fold=1
    epochs = 25
    batch_size = 32*2
    log_interval = 32*8
    data_path = 'Data_RFSG'
    
    device = torch.device('cuda')


    data = []
    lengths = []
    folders = os.listdir(data_path)
    for folder in folders:

        features_v = numpy.load(os.path.join(data_path, folder, folder + '_VIDEO_' + feature_type + '_ALL_FEATURES_' + label_type + '.npy'))
        features_a = numpy.load(os.path.join(data_path, folder, folder + '_AUDIO_' + feature_type + '_ALL_FEATURES_' + label_type + '.npy'))
        features = numpy.stack((features_v, features_a), axis=-1)

        labels = numpy.load(os.path.join(data_path, folder, folder + '_ALL_LABELS_' + label_type + '.npy'))

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        dataset = torch.utils.data.TensorDataset(features, labels)
        lengths.append(len(dataset))
        data.append(dataset)


    dataset = torch.utils.data.ConcatDataset(data)
    class_weights = calculate_class_weights(dataset, 2)
    print(class_weights)

    # cross_validator = KFold(n_splits=num_folds, shuffle=True, random_state=101)
    cross_validator_splits = custom_l6o_iterator(lengths) #cross_validator.split(range(len(dataset)))

    for train_idx, val_idx, test_idx in cross_validator_splits:
        print("Fold: ",fold)
        # device = torch.device('cuda')

        if model_type == "TCN":
            model = Model_TCN(num_layers).to(device)
        else:
            model = Model_BLSTM(num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

        # train_len = int(len(train_idx) * 0.9)

        # val_idx = train_idx[train_len:]
        # train_idx = train_idx[:train_len]

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        best_f1 = 0
        counter = 0
        train_metrics = {"f1s":[], "aurocs":[], "mAPs":[]}
        val_metrics = {"f1s":[], "aurocs":[], "mAPs":[]}

        for epoch in range(1, epochs + 1):
            if counter > patience: continue
            print('Train', epoch, flush=True)
            f1s, aurocs, mAPs = train(model, device, train_loader, optimizer, criterion, log_interval, epoch, model_type)
            train_metrics["f1s"] += f1s
            train_metrics["aurocs"] += aurocs
            train_metrics["mAPs"] += mAPs

            print('Validate', epoch, flush=True)
            acc, f1, auroc, mAP = validate(model, device, val_loader, criterion, epoch, model_type)
            val_metrics["f1s"] += [f1]
            val_metrics["aurocs"] += [auroc]
            val_metrics["mAPs"] += [mAP]
            
            if f1 > best_f1:
                best_f1 = f1
                counter=0
                
                print('\33[31m\tSaving new best model...\33[0m')
                os.makedirs('checkpoints', exist_ok=True)
                state = {'epoch': epoch, 'model': model.state_dict()}
                torch.save(state, f'{directory}/{fold}-fold_model.pth')
            else: counter +=1

            print("Plot Figures")
            for k,v in train_metrics.items():
                plt.plot(v, label=k)

            plt.xticks(range(epochs), range(1, epochs + 1))
            plt.legend()
            plt.savefig(f"{directory}/{fold}-fold-train.png")
            plt.clf()

            for k2,v2 in val_metrics.items():
                plt.plot(v2, label=k2)

            plt.xticks(range(epochs), range(1, epochs + 1))
            plt.legend()
            plt.savefig(f"{directory}/{fold}-fold-val.png")
            plt.clf()
    

        fold += 1


    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # num_folds=11
    # patience=10
    # fold=1
    # epochs = 25
    # batch_size = 64
    # log_interval = 32*8
    # data_path = 'Data_RFSG'
    
    # device = torch.device('cuda')

    # #Note - make sure to update the checkpoint dir so that you don't overwrite your saved models!!!
    # # I have 69 people so I am doing 11 folds

    # folders = os.listdir(data_path)

    # cross_validator = KFold(n_splits=num_folds, shuffle=False)#, random_state=101)
    # cross_validator_splits = cross_validator.split(range(len(folders)))

    # for train_fldr_idxs, _ in cross_validator_splits:
    #     print("Fold: ",fold)
    #     print("Including: ", train_fldr_idxs)

    #     data = []
    #     for idx, folder in enumerate(folders):
    #         if idx in train_fldr_idxs:

    #             features_v = numpy.load(os.path.join(data_path, folder, folder + '_VIDEO_' + feature_type + '_ALL_FEATURES_' + label_type + '.npy'))
    #             features_a = numpy.load(os.path.join(data_path, folder, folder + '_AUDIO_' + feature_type + '_ALL_FEATURES_' + label_type + '.npy'))
    #             features = numpy.stack((features_v, features_a), axis=-1)

    #             labels = numpy.load(os.path.join(data_path, folder, folder + '_ALL_LABELS_' + label_type + '.npy'))

    #             features = torch.FloatTensor(features)
    #             labels = torch.LongTensor(labels)

    #             dataset = torch.utils.data.TensorDataset(features, labels)
    #             data.append(dataset)


    #     dataset = torch.utils.data.ConcatDataset(data)
    #     class_weights = calculate_class_weights(dataset, 2)
    #     print(class_weights)

    #     if model_type == "TCN":
    #         model = Model_TCN(num_layers).to(device)
    #     else:
    #         model = Model_BLSTM(num_layers).to(device)
            
    #     optimizer = torch.optim.Adam(model.parameters())
    #     criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    #     train_idx = list(range(len(dataset)))
    #     random.shuffle(train_idx)
    #     print("sample train indx:", train_idx[:10])

    #     train_len = int(len(train_idx) * 0.9)

    #     val_idx = train_idx[train_len:]
    #     train_idx = train_idx[:train_len]

    #     print("Check if lists are seperate")
    #     check = list_contains(val_idx, train_idx)
    #     print("Yes matches?", check)
    #     input("Continue?")

    #     train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    #     val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    #     best_f1 = 0
    #     counter = 0
    #     train_metrics = {"f1s":[], "aurocs":[], "mAPs":[]}
    #     val_metrics = {"f1s":[], "aurocs":[], "mAPs":[]}

    #     for epoch in range(1, epochs + 1):
    #         if counter > patience: continue
    #         print('Train', epoch, flush=True)
    #         f1s, aurocs, mAPs = train(model, device, train_loader, optimizer, criterion, log_interval, epoch, model_type)
    #         train_metrics["f1s"] += f1s
    #         train_metrics["aurocs"] += aurocs
    #         train_metrics["mAPs"] += mAPs

    #         print('Validate', epoch, flush=True)
    #         acc, f1, auroc, mAP = validate(model, device, val_loader, criterion, epoch, model_type)
    #         val_metrics["f1s"] += [f1]
    #         val_metrics["aurocs"] += [auroc]
    #         val_metrics["mAPs"] += [mAP]
            
    #         if f1 > best_f1:
    #             best_f1 = f1
    #             counter=0
                
    #             print('\33[31m\tSaving new best model...\33[0m')
    #             os.makedirs('checkpoints', exist_ok=True)
    #             state = {'epoch': epoch, 'model': model.state_dict()}
    #             torch.save(state, f'{directory}/{fold}-fold_model.pth')
    #         else: counter +=1

    #         print("Plot Figures")
    #         for k,v in train_metrics.items():
    #             plt.plot(v, label=k)

    #         plt.xticks(range(epochs), range(1, epochs + 1))
    #         plt.legend()
    #         plt.savefig(f"{directory}/{fold}-fold-train.png")
    #         plt.clf()

    #         for k2,v2 in val_metrics.items():
    #             plt.plot(v2, label=k2)

    #         plt.xticks(range(epochs), range(1, epochs + 1))
    #         plt.legend()
    #         plt.savefig(f"{directory}/{fold}-fold-val.png")
    #         plt.clf()
    

    #     fold += 1


if __name__ == '__main__':
    trainer = "chris"
    layers = 2
    for features in ["PERFECTMATCH","SYNCNET"]:
        for model in ["BLSTM"]:# "BLSTM","TCN"
            for label in ["TURN", "SPEECH"]:
                print(model,features,label)
                main(model,features,label,layers, trainer=trainer)