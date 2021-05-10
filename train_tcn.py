import os
import torch

import numpy

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tcn import TemporalConvNet

class Model_TCN(torch.nn.Module):
    def __init__(self):
        super(Model_TCN, self).__init__()
        
        self.nhid = 128
        
        self.tcn_v = TemporalConvNet(512, [self.nhid, self.nhid, self.nhid, self.nhid], 3, dropout=0.3)
        self.tcn_a = TemporalConvNet(512, [self.nhid, self.nhid, self.nhid, self.nhid], 3, dropout=0.3)
        self.fc = torch.nn.Linear(2*self.nhid, 2)

    def forward(self, x_v, x_a):
        x_v = self.tcn_v(x_v)
        x_a = self.tcn_a(x_a)
        x_v = x_v[:, :, -1]
        x_a = x_a[:, :, -1]
        
        x = self.fc(torch.cat((x_v, x_a), axis=1))
        
        return x

def train(model, device, train_loader, optimizer, criterion, log_interval, epoch):
    model.train()

    predictions, targets = [], []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data_v, data_a = data[:,:,:,0], data[:,:,:,1]
        
        data_v, data_a, target = torch.FloatTensor(data_v).to(device), torch.FloatTensor(data_a).to(device), torch.LongTensor(target).to(device)
        data_v, data_a, target = Variable(data_v), Variable(data_a), Variable(target)
        
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

        # print(targets, predictions)
        # print(target, pred)
        # print(target.tolist(), pred.tolist())

        
        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('\tEpoch {} [{}/{} ({:.0f}%)]\tLoss {:.5f} \tF1: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), f1_score(targets, predictions)))
            # print(targets, "\n\n", predictions)
            # print("F1:", f1_score(targets, predictions))

def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    
    loss = 0
    correct = 0
    targets, predictions = [], []

    
    with torch.no_grad():
        for data, target in val_loader:
            data_v, data_a = data[:,:,:,0], data[:,:,:,1]
            
            data_v, data_a, target = torch.FloatTensor(data_v).to(device), torch.FloatTensor(data_a).to(device), torch.LongTensor(target).to(device)
            data_v, data_a, target = Variable(data_v), Variable(data_a), Variable(target)
            
            data_v = data_v.permute(0, 2, 1)
            data_a = data_a.permute(0, 2, 1)
        
            output = model(data_v, data_a)
            
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            # predictions += pred
            # targets += target
            correct += pred.eq(target.view_as(pred)).sum().item()

            pred = pred.detach().cpu()
            predictions += pred.tolist()

            target = target.cpu().view_as(pred)
            targets += target.tolist()
    
    loss /= len(val_loader.dataset)
    # alt_acc = 100. * predictions.eq(targets.view_as(predictions)).sum().item() / len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)

    # print("alt_acc", alt_acc)

    print('\tLoss {:.4f}\tAccuracy {}/{} ({:.0f}%)'.format(
        loss, correct, len(val_loader.dataset), acc))
    print("F1: ", f1_score(targets, predictions))
        
    return acc

def main():
    epochs = 100
    batch_size = 32*2
    log_interval = 32*6
    data_path = 'data'
    feature_type = 'SYNCNET'
    
    device = torch.device('cuda')

    model = Model_TCN().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    train_data = []
    val_data = []

    folders = os.listdir(data_path)
    for folder in folders:
        features_v = numpy.load(os.path.join(data_path, folder, folder + '_VIDEO_' + feature_type + '_FEATURES.npy'))
        features_a = numpy.load(os.path.join(data_path, folder, folder + '_AUDIO_' + feature_type + '_FEATURES.npy'))
        features = numpy.stack((features_v, features_a), axis=-1)
     
        labels = numpy.load(os.path.join(data_path, folder, folder + '_LABELS.npy'))
        
        train_idx, val_idx, _, _ = train_test_split(range(len(labels)), labels, shuffle=False, test_size=0.1, random_state=101)
        
        train_features = torch.FloatTensor(features[train_idx])
        train_labels = torch.LongTensor(labels[train_idx])
        
        val_features = torch.FloatTensor(features[val_idx])
        val_labels = torch.LongTensor(labels[val_idx])
        
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_data.append(train_dataset)
        
        val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
        val_data.append(val_dataset)
    
    train_dataset = torch.utils.data.ConcatDataset(train_data)
    val_dataset = torch.utils.data.ConcatDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    best_acc = 0
    for epoch in range(1, epochs + 1):
        print('Train')
        train(model, device, train_loader, optimizer, criterion, log_interval, epoch)
        
        print('Validate')
        acc = validate(model, device, val_loader, criterion, epoch)
        
        if acc > best_acc:
            best_acc = acc
            
            print('\33[31m\tSaving new best model...\33[0m')
            os.makedirs('checkpoints', exist_ok=True)
            state = {'epoch': epoch, 'model': model.state_dict(), 'acc': acc}
            torch.save(state, 'checkpoints/TCN_' + feature_type + '.pth')

if __name__ == '__main__':
    main()