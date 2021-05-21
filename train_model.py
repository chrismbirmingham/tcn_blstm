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
            # print('\tEpoch {} [{}/{} ({:.0f}%)]\tLoss {:.4f}\tF1 {:.3f}\tauROC {:.3f}\tmAP {:.3f}'.format(
            #     epoch, 
            #     batch_idx * len(data), 
            #     len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), 
            #     loss.item(),
            #     f1, auroc, mAP)
            # )
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

def main(model_type, feature_type, label_type, num_layers, trainer="chris"):
    directory = f"checkpoints/{trainer}/{num_layers}LAYER/{label_type}/{model_type}_{feature_type}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    num_folds=10
    epochs = 20
    batch_size = 32
    log_interval = 32*10
    data_path = 'data'
    
    device = torch.device('cuda')

    if model_type == "TCN":
        model = Model_TCN(num_layers).to(device)
    else:
        model = Model_BLSTM(num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for fold in range(10):
        print("Fold: ", fold)

        train_data = []
        val_data = []
        test_data = []

        folders = os.listdir(data_path)
        for folder in folders:
            # We will be using the val set as the test set and cross validating on the train set
            train_features_v = numpy.load(os.path.join(data_path, folder, f"{folder}_VIDEO_{feature_type}_TRAIN_FEATURES_{label_type}.npy"))
            train_features_a = numpy.load(os.path.join(data_path, folder, f"{folder}_AUDIO_{feature_type}_TRAIN_FEATURES_{label_type}.npy"))
            train_features = numpy.stack((train_features_v, train_features_a), axis=-1)
        
            train_labels = numpy.load(os.path.join(data_path, folder, f"{folder}_TRAIN_LABELS_{label_type}.npy"))
            
            train_features = torch.FloatTensor(train_features)
            train_labels = torch.LongTensor(train_labels)
            
            training_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
            
            fold_size = int(len(training_dataset)/num_folds)
            val_fold_start = fold*fold_size
            val_fold_end = (fold+1)*fold_size

            val_fold_indices = list(range(val_fold_start, val_fold_end))
            val_dataset = torch.utils.data.Subset(training_dataset, val_fold_indices)

            train_fold_indices = list(range(0,val_fold_start)) + list(range(val_fold_end, len(training_dataset)))
            train_dataset = torch.utils.data.Subset(training_dataset, train_fold_indices)

            val_data.append(val_dataset)
            train_data.append(train_dataset)
            
            test_features_v = numpy.load(os.path.join(data_path, folder, f"{folder}_VIDEO_{feature_type}_VAL_FEATURES_{label_type}.npy"))
            test_features_a = numpy.load(os.path.join(data_path, folder, f"{folder}_AUDIO_{feature_type}_VAL_FEATURES_{label_type}.npy"))
            test_features = numpy.stack((test_features_v, test_features_a), axis=-1)
        
            test_labels = numpy.load(os.path.join(data_path, folder, f"{folder}_VAL_LABELS_{label_type}.npy"))
            
            test_features = torch.FloatTensor(test_features)
            test_labels = torch.LongTensor(test_labels)
            
            test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
            test_data.append(test_dataset)


        train_dataset = torch.utils.data.ConcatDataset(train_data)
        val_dataset = torch.utils.data.ConcatDataset(val_data)
        test_dataset = torch.utils.data.ConcatDataset(test_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        best_f1 = 0
        train_metrics = {"f1s":[], "aurocs":[], "mAPs":[]}
        val_metrics = {"f1s":[], "aurocs":[], "mAPs":[]}

        for epoch in range(1, epochs + 1):
            print('Train')
            f1s, aurocs, mAPs = train(model, device, train_loader, optimizer, criterion, log_interval, epoch, model_type)
            train_metrics["f1s"] += f1s
            train_metrics["aurocs"] += aurocs
            train_metrics["mAPs"] += mAPs

            print('Validate')
            acc, f1, auroc, mAP = validate(model, device, val_loader, criterion, epoch, model_type)
            val_metrics["f1s"] += [f1]
            val_metrics["aurocs"] += [auroc]
            val_metrics["mAPs"] += [mAP]

            print("plot")
            for k,v in train_metrics.items():
                plt.plot(v, label=k)
            plt.legend()
            plt.savefig(f"{directory}/{fold}-fold-train.png")
            plt.clf()
            for k2,v2 in val_metrics.items():
                plt.plot(v2, label=k2)
            plt.legend()
            plt.savefig(f"{directory}/{fold}-fold-val.png")
            plt.clf()
            
            if f1 > best_f1:
                best_f1 = f1
                
                print('\33[31m\tSaving new best model...\33[0m')
                os.makedirs('checkpoints', exist_ok=True)
                state = {'epoch': epoch, 'model': model.state_dict()}
                torch.save(state, f'{directory}/{fold}-fold_model.pth')
        
        # Testing
        if model_type == "TCN":
            model = Model_TCN(num_layers).to(device)
        else:
            model = Model_BLSTM(num_layers).to(device)
        model.load_state_dict(torch.load(f'{directory}/{fold}-fold_model.pth')["model"])
        test_metrics={}
        acc, f1, auroc, mAP = validate(model, device, test_loader, criterion, epoch, model_type)
        test_metrics["acc"] = acc
        test_metrics["f1"] = f1
        test_metrics["auroc"] = auroc
        test_metrics["mAP"] = mAP
        with open(f'{directory}/{fold}-fold_test_scores.json', 'w') as f:

            json = json.dumps(test_metrics)
            f.write(json)
            f.close()



if __name__ == '__main__':
    features = "PERFECTMATCH"
    trainer = "chris"
    layers = 2
    for model in ["BLSTM"]:# "TCN",
        for label in ["SPEECH", "TURN"]:
            print(model,features,label)
            main(model,features,label,layers, trainer=trainer)