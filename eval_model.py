import os
import torch

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
        
        self.lstm_v = torch.nn.LSTM(512, self.nhid, num_layers=2, bidirectional=True)
        self.lstm_a = torch.nn.LSTM(512, self.nhid, num_layers=2, bidirectional=True)
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
        
        self.tcn_v = TemporalConvNet(512, [self.nhid, self.nhid], 3, dropout=0.3)
        self.tcn_a = TemporalConvNet(512, [self.nhid, self.nhid], 3, dropout=0.3)
        self.fc = torch.nn.Linear(2*self.nhid, 2)

    def forward(self, x_v, x_a):
        x_v = self.tcn_v(x_v)
        x_a = self.tcn_a(x_a)
        x_v = x_v[:, :, -1]
        x_a = x_a[:, :, -1]
        
        x = self.fc(torch.cat((x_v, x_a), axis=1))
        
        return torch.nn.functional.log_softmax(x, dim=1)


def validate(model, device, val_loader, model_type="TCN"):
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
    try:
        auroc = roc_auc_score(targets, outputs1)
    except Exception as e:
        print(e)
        auroc=.5
    AP1 = average_precision_score(targets, outputs1)
    AP0 = average_precision_score(list(1-numpy.array(targets)), outputs0)
    mAP = (AP1 + AP0)/2

    print('\tLoss {:.5f}\tAccuracy {:.3f}\tF1: {:.3f}\tauROC: {:.3f}\tmAP: {:.3f}'.format(
        loss, acc, f1, auroc, mAP))
        
    return acc, f1, auroc, mAP, outputs0, outputs1

def main(model_type, feature_type, label_type, model_trainer=""):
    print((model_type, feature_type))
    batch_size = 32
    data_path = 'data'
    
    device = torch.device('cuda')

    if model_type == "TCN":
        model = Model_TCN().to(device)
    else:
        model = Model_BLSTM().to(device)

    model.load_state_dict(torch.load(f'{model_trainer}checkpoints/{model_type}_{feature_type}_{label_type}.pth')["model"])

    all_data = []

    folders = os.listdir(data_path)
    for folder in folders:
        print(folder)
        all_features_v = numpy.load(os.path.join(data_path, folder, f"{folder}_VIDEO_{feature_type}_ALL_FEATURES_{label_type}.npy"))
        all_features_a = numpy.load(os.path.join(data_path, folder, f"{folder}_AUDIO_{feature_type}_ALL_FEATURES_{label_type}.npy"))

        all_features = numpy.stack((all_features_v, all_features_a), axis=-1)
     
        all_labels = numpy.load(os.path.join(data_path, folder, f"{folder}_ALL_LABELS_{label_type}.npy"))
        
        all_features = torch.FloatTensor(all_features)
        all_labels = torch.LongTensor(all_labels)
        
        all_dataset = torch.utils.data.TensorDataset(all_features, all_labels)


        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        acc, f1, auroc, mAP, out0, out1 = validate(model, device, all_loader, model_type=model_type)

        model_out = zip(out0,out1)
        df = pandas.DataFrame(model_out, columns=["0Conf","1Conf"])
        df.to_csv(os.path.join(data_path, folder, f"{folder}_{model_type}_{feature_type}_{label_type}_conf{model_trainer}.csv"))


if __name__ == '__main__':
    for m in ["BLSTM"]:#"TCN",
        for f in ["SYNCNET","PERFECTMATCH"]:
            for l in ["SPEECH"]:#, "TURN"
                print(m,f,l)
                main(m,f,l, model_trainer="kalin")