import os
import torch

import numpy

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

class Model_BLSTM(torch.nn.Module):
    def __init__(self):
        super(Model_BLSTM, self).__init__()
        
        self.nhid = 128
        
        self.lstm_v = torch.nn.LSTM(512, self.nhid, bidirectional=True)
        self.lstm_a = torch.nn.LSTM(512, self.nhid, bidirectional=True)
        self.fc = torch.nn.Linear(2*self.nhid*2*5, 2)

    def forward(self, x_v, x_a):
        x_v = self.lstm_v(x_v)[0]
        x_a = self.lstm_a(x_a)[0]
        x_v = x_v.view(-1, self.nhid*2*5)
        x_a = x_a.view(-1, self.nhid*2*5)
        
        x = self.fc(torch.cat((x_v, x_a), axis=1))
        
        return x

def train(model, device, train_loader, optimizer, criterion, log_interval, epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data_v, data_a = data[:,:,:,0], data[:,:,:,1]
        
        data_v, data_a, target = torch.FloatTensor(data_v).to(device), torch.FloatTensor(data_a).to(device), torch.LongTensor(target).to(device)
        data_v, data_a, target = Variable(data_v), Variable(data_a), Variable(target)
        
        output = model(data_v, data_a)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('\tEpoch {} [{}/{} ({:.0f}%)]\tLoss {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    
    loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data_v, data_a = data[:,:,:,0], data[:,:,:,1]
            
            data_v, data_a, target = torch.FloatTensor(data_v).to(device), torch.FloatTensor(data_a).to(device), torch.LongTensor(target).to(device)
            data_v, data_a, target = Variable(data_v), Variable(data_a), Variable(target)
        
            output = model(data_v, data_a)
            
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)

    print('\tLoss {:.4f}\tAccuracy {}/{} ({:.0f}%)'.format(
        loss, correct, len(val_loader.dataset), acc))
        
    return acc

def main():
    epochs = 100
    batch_size = 32
    log_interval = 32*5
    data_path = 'data'
    feature_type = 'PERFECTMATCH'
    
    device = torch.device('cuda')

    model = Model_BLSTM().to(device)
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
        
        train_idx, val_idx, _, _ = train_test_split(range(len(labels)), labels, test_size=0.1, random_state=101, stratify=labels)
        
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
            torch.save(state, 'checkpoints/BLSTM_' + feature_type + '.pth')

if __name__ == '__main__':
    main()