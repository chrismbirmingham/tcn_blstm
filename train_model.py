import os
import torch
from tqdm import tqdm
import numpy
import pandas
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from tcn import TemporalConvNet

class Model_BLSTM(torch.nn.Module):
    def __init__(self):
        super(Model_BLSTM, self).__init__()
        
        self.nhid = 128
        
        self.lstm_v = torch.nn.LSTM(512, self.nhid, num_layers=1, bidirectional=True)
        self.lstm_a = torch.nn.LSTM(512, self.nhid, num_layers=1, bidirectional=True)
        self.fc = torch.nn.Linear(2*self.nhid*2*5, 2)

    def forward(self, x_v, x_a):
        x_v = self.lstm_v(x_v)[0]
        x_a = self.lstm_a(x_a)[0]
        x_v = x_v.view(-1, self.nhid*2*5)
        x_a = x_a.view(-1, self.nhid*2*5)
        
        x = self.fc(torch.cat((x_v, x_a), axis=1))
        
        return torch.nn.functional.log_softmax(x, dim=1)

class Model_TCN(torch.nn.Module):
    def __init__(self):
        super(Model_TCN, self).__init__()
        
        self.nhid = 128
        
        self.tcn_v = TemporalConvNet(512, [self.nhid], 3, dropout=0.3)
        self.tcn_a = TemporalConvNet(512, [self.nhid], 3, dropout=0.3)
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

def main(model_type, feature_type, label_type):
    epochs = 20
    batch_size = 32
    log_interval = 32*10
    data_path = 'data'
    
    device = torch.device('cuda')

    if model_type == "TCN":
        model = Model_TCN().to(device)
    else:
        model = Model_BLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    train_data = []
    val_data = []

    folders = os.listdir(data_path)
    for folder in folders:
        train_features_v = numpy.load(os.path.join(data_path, folder, f"{folder}_VIDEO_{feature_type}_TRAIN_FEATURES_{label_type}.npy"))
        train_features_a = numpy.load(os.path.join(data_path, folder, f"{folder}_AUDIO_{feature_type}_TRAIN_FEATURES_{label_type}.npy"))
        train_features = numpy.stack((train_features_v, train_features_a), axis=-1)
     
        train_labels = numpy.load(os.path.join(data_path, folder, f"{folder}_TRAIN_LABELS_{label_type}.npy"))
        
        val_features_v = numpy.load(os.path.join(data_path, folder, f"{folder}_VIDEO_{feature_type}_VAL_FEATURES_{label_type}.npy"))
        val_features_a = numpy.load(os.path.join(data_path, folder, f"{folder}_AUDIO_{feature_type}_VAL_FEATURES_{label_type}.npy"))
        val_features = numpy.stack((val_features_v, val_features_a), axis=-1)
     
        val_labels = numpy.load(os.path.join(data_path, folder, f"{folder}_VAL_LABELS_{label_type}.npy"))
        
        train_features = torch.FloatTensor(train_features)
        train_labels = torch.LongTensor(train_labels)
        
        val_features = torch.FloatTensor(val_features)
        val_labels = torch.LongTensor(val_labels)
        
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_data.append(train_dataset)
        
        val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
        val_data.append(val_dataset)
    
    train_dataset = torch.utils.data.ConcatDataset(train_data)
    val_dataset = torch.utils.data.ConcatDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

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
        plt.savefig(f"{model_type}_{feature_type}_{label_type}-train.png")
        plt.clf()
        for k2,v2 in val_metrics.items():
            plt.plot(v2, label=k2)
        plt.legend()
        plt.savefig(f"{model_type}_{feature_type}_{label_type}-val.png")
        plt.clf()
        
        if f1 > best_f1:
            best_f1 = f1
            
            print('\33[31m\tSaving new best model...\33[0m')
            os.makedirs('checkpoints', exist_ok=True)
            state = {'epoch': epoch, 'model': model.state_dict()}
            torch.save(state, f'checkpoints/{model_type}_{feature_type}_{label_type}-small.pth')

if __name__ == '__main__':
    for m in ["TCN","BLSTM"]:
        for f in ["SYNCNET","PERFECTMATCH"]:
            for l in ["SPEECH", "TURN"]:
                print(m,f,l)
                main(m,f,l)