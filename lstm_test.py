import argparse
import sys
import train
import transformer

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

class janossyLastOnlyLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim, janossy_count, dropout=0):
        super(janossyLastOnlyLSTM, self).__init__()
        self.janossy_count = janossy_count
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, dropout = dropout)
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, seq, prin=False):
        # lstm needs [seq_len, batch, input_size]
        seq = seq.transpose(0, 1)
        seq_len, batch_len, input_dim = seq.size()
        
        indices = np.arange(seq_len)
        reverse_indices = np.arange(seq_len)
        indices_arange = np.arange(seq_len)
        
        result = torch.zeros([seq_len, batch_len, self.hidden_dim], device=seq.device)
        for _ in range(self.janossy_count):
            for i in range(len(indices)):
                np.random.shuffle(indices)
                ind = np.where(indices==i)[0][0]
                indices[[ind, -1]] = indices[[-1, ind]]
                reverse_indices[indices] = indices_arange
                
                permuted_input = seq[indices]
                
#                 print(permuted_input)
                _, (last_h, _) = self.lstm(permuted_input)
                result[i] += last_h.squeeze(0)
#                 print(permuted_input.shape)

        result = result / self.janossy_count
        # get result of shape [batch, seq_len, hidden_size]
        result = result.transpose(0, 1)
        result = self.w2(F.relu(self.w1(result)))
#         if prin:
#             print('res:', result[0,:2])
#         result = torch.log((result+1)/(1-result))
#         if prin:
#             print('resA:', result[0,:2])
        return result
# janossyLastOnlyLSTM(4, 4, 1)(torch.arange(64, dtype=torch.float).reshape(4, 4, 4))



def get_trainloader_test():
    def create_data(num_examples, seq_len, n_dim, std):
        train_x = torch.normal(torch.zeros(seq_len*n_dim*num_examples), torch.ones(seq_len*n_dim*num_examples)*std).view(num_examples, seq_len, n_dim)
        means = torch.mean(train_x, dim=2)
        maxes = torch.max(train_x, dim=2).values
        sum_of_means = means.sum(dim=1)
        train_y = maxes + sum_of_means[:, None] - means
    #     train_y = maxes
        return train_x, train_y.unsqueeze(-1)

    class MyDataset(Dataset):
        def __init__(self, x, y):
            super(MyDataset, self).__init__()
            assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
            self.x = x
            self.y = y


        def __len__(self):
            return self.y.shape[0]

        def __getitem__(self, index):
            return self.x[index], self.y[index]


    seq_len = 10
    n_dim = 5
    num_examples = 1024*10
    std = 1
    train_x, train_y = create_data(num_examples, seq_len, n_dim, std)
    traindata = MyDataset(train_x, train_y)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=1024, shuffle=True)
    return trainloader






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_model', action='store_true')
    parser.add_argument('-epoch', type=int, default=1e5)
    opt = parser.parse_args()
    
        

    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    # device = 'cpu'

    trainloader = get_trainloader_test()
    trainloader.dataset.x = trainloader.dataset.x.to(device)
    trainloader.dataset.y = trainloader.dataset.y.to(device)
    n_dim = trainloader.dataset.x.size()[-1]
    # model = janossyLSTM(n_dim, 1, janossy_count=2)

    model = janossyLastOnlyLSTM(input_dim=n_dim, hidden_dim=10, out_dim=1, janossy_count=1).to(device)
    cur_epoch = 1
    if opt.load_model:
        print('[ Loading Model ]')
        checkpoint = torch.load('train_lstm_test.chkpt')
        model.load_state_dict(checkpoint['model'])
        cur_epoch = checkpoint['epoch']
        print('[ Epochs 1 -', cur_epoch, 'Done ]')
        
    # model = transformer.Transformer_embed_ready_test(d_model=5, d_inner=100, n_layers=1, n_head=1, d_k=4, d_v=4, dropout=0)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-04)
    
    print('epoch: ', end=' ')
    for epoch in range(cur_epoch, int(opt.epoch)+1):
        print(epoch, end=' ', flush=True)
        for i, (mini_x, mini_y) in enumerate(trainloader):
            model.zero_grad()
            out = model(mini_x, prin=epoch%5==0 and i==0)

            loss = loss_function(out, mini_y)
            loss.backward()
            optimizer.step()
            if epoch%20==0 and i==0:
                print()
                print('loss:', loss.item())
                print(' y :', mini_y[:1, :5].squeeze().data.cpu().numpy())
                print('out:', out[:1, :5].squeeze().data.cpu().numpy())
                print('dlt:', (out[:1, :5]-mini_y[:1, :5]).squeeze().data.cpu().numpy())
                print(' % :', ((out[:1, :5]/mini_y[:1, :5] - 1)*100).squeeze().data.cpu().numpy())
                print('-----------')
                torch.save({'model': model.state_dict(), 'epoch': epoch}, 'train_lstm_test.chkpt')
                # print(out[0][0], mini_y[0, 0])
                print('epoch: ', end=' ')
    print()
    print('------')
    print('Done')