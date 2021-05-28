import os
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
from sklearn.model_selection import KFold
from tcn import TemporalConvNet

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
    directory = f"checkpoints/{trainer}/{num_layers}LAYER/{label_type}/{model_type}_{feature_type}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    num_folds=10
    fold=1
    batch_size = 64
    data_path = 'Data_RFSG'
    
    device = torch.device('cuda')


    data = []
    folders = os.listdir(data_path)
    for folder in folders:

        features_v = numpy.load(os.path.join(data_path, folder, folder + '_VIDEO_' + feature_type + '_ALL_FEATURES_' + label_type + '.npy'))
        features_a = numpy.load(os.path.join(data_path, folder, folder + '_AUDIO_' + feature_type + '_ALL_FEATURES_' + label_type + '.npy'))
        features = numpy.stack((features_v, features_a), axis=-1)

        labels = numpy.load(os.path.join(data_path, folder, folder + '_ALL_LABELS_' + label_type + '.npy'))

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        dataset = torch.utils.data.TensorDataset(features, labels)
        data.append(dataset)


    dataset = torch.utils.data.ConcatDataset(data)
    class_weights = calculate_class_weights(dataset, 2)
    print(class_weights)

    cross_validator = KFold(n_splits=num_folds, shuffle=True, random_state=101)
    cross_validator_splits = cross_validator.split(range(len(dataset)))

    for train_idx, test_idx in cross_validator_splits:
        print("Fold: ",fold)
        # device = torch.device('cuda')

        if model_type == "TCN":
            model = Model_TCN(num_layers).to(device)
        else:
            model = Model_BLSTM(num_layers).to(device)
        model.load_state_dict(torch.load(f'{directory}/{fold}-fold_model.pth')["model"])

        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
        
        test_metrics={}
        print("Test",0)
        acc, f1, auroc, mAP = validate(model, device, test_loader, criterion, 0, model_type)
        test_metrics["acc"] = acc
        test_metrics["f1"] = f1
        test_metrics["auroc"] = auroc
        test_metrics["mAP"] = mAP
        with open(f'{directory}/{fold}-fold_test_scores.json', 'w') as f:

            json_obj = json.dumps(test_metrics)
            f.write(json_obj)
            f.close()

        fold += 1


if __name__ == '__main__':
    # features = "PERFECTMATCH"
    trainer = "chris"
    layers = 2
    for model in ["TCN","BLSTM"]:# "BLSTM","TCN"
        for features in ["PERFECTMATCH","SYNCNET"]:
            for label in ["TURN", "SPEECH"]:
                print(model,features,label)
                main(model,features,label,layers, trainer=trainer)