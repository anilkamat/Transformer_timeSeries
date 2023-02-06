from turtle import forward
from numpy import size
from statsmodels import test
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
from Utils.utils import *
from dataloader import dataloader 
from plotter import *
from tqdm import tqdm

# TODO Code for positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, max_length, dim_feature): # max_length-> seq_length, dim_feat -> num_features
        super(PositionalEncoding, self).__init__()
        self.dim_feature = dim_feature
        self.max_length = max_length

        PE_matrix = torch.zeros((self.max_length, self.dim_feature))
        for pos in range(self.max_length):
            for i in range(int(dim_feature/2)):
                denominator =  np.power(100,2*i/dim_feature)
                PE_matrix[pos,2*i] = np.sin(pos/ denominator)
                PE_matrix[pos,2*i+1] = np.cos(pos/ denominator)
        self.register_buffer('PE_matrix', PE_matrix)

    def forward(self, x):
        # print('self.PE_matrix :',self.PE_matrix.shape, x.shape)
        r = x+self.PE_matrix[:x.size(0), :] #self.PE_matrix
        return r
#TODO Code of model
class model_transformer(nn.Module):
    def __init__(self,  in_channels, out_channels, n_layers, n_heads, h1size, h2size, dropout_):
        super(model_transformer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.h1_size = h1size
        self.h2_size = h2size
        self.in_channels = in_channels
        self.out_channels = out_channels
        feature_size = 32
        seq_len =8
        dropout = dropout_

        self.PE_x = PositionalEncoding(seq_len, feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=self.n_heads, 
            dim_feedforward=self.h1_size,batch_first=True,dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)
        
        self.PE_y = PositionalEncoding(1, feature_size)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_size, nhead=self.n_heads,
         dim_feedforward= self.h1_size, batch_first=True
         )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        # self.transform1 = nn.Transformer(d_model=self.in_channels, nhead = self.n_heads, num_encoder_layers=self.n_layers
        # ,num_decoder_layers=n_layers, dim_feedforward=self.h1_size, batch_first=True)
        self.linear1 = nn.Linear(feature_size, self.out_channels)
        # self.linear2 = nn.Linear(self.h2_size, out_channels)

    def forward(self, x,y):
        #print(x.shape, y.shape)
        out = self.PE_x(x)
        #print('PE out.shape', out.shape)
        out = self.encoder(out)
        #print('encoder out.shape', out.shape)
        y = self.PE_y(y)
        #print('PE_dec out.shape', y.shape)
        out = self.decoder(y, out)
        #print('dec out.shape', out.shape)
        #out = self.transform1(x,y)
        out = self.linear1(out).squeeze()
        #print('linear out.shape', out.shape)
        # out = self.linear2(out)
        
        return out

model = model_transformer(in_channels=1, out_channels=1, n_layers=2, n_heads=1, h1size=4, h2size=8, dropout_=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
loss_fn = nn.MSELoss()
# TODO simulated data to test the model
## generate a simple random dataset to test the model
n_hist = 8
n_pred = 1
batch_size = 32
x = torch.linspace(0,2,2000)
y = 2*torch.sin(60*torch.pi*x)  +torch.randn(len(x))
y,min,max = min_max_normalization(y)
# print('y.shape ',y.shape)
num_time_series = 1
plot_signals(y,1)
# y = split_timeseries(y, num_time_series)

src = torch.randn((32,8,1)) # (batch_size, input_sequence, num_features)
trg = torch.randn((32,1,1))
# yy = model(src,trg).squeeze()
losses_epoch = []
losses_epoch_eval = []
x_eval = []
train_loader, test_loader = dataloader(y.T,n_hist, n_pred, batch_size, split_size=0.8, shuffle = False)


for epoch in range(200):
    loop = tqdm(train_loader)
    losses_batch = []
    model.train()
    for batch_index, (src, trg) in enumerate(loop):
        # print(src.shape, trg.shape)
        # print('batch index: ', src.shape)
        yy = model(src,trg)
        #print('yy: \n',yy)
        loss = torch.sqrt(loss_fn(yy,trg.squeeze()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_batch.append(loss.detach().numpy())
    losses_epoch.append(sum(losses_batch)/(batch_index+1))
    print(f'{epoch } train loss : {sum(losses_batch)/(batch_index+1)}')
    
    losses_batch_eval = []
    if epoch%5 ==0:
        model.eval()
        for batch_index, (src, trg) in enumerate(test_loader):
            # print(src.shape, trg.shape)
            # print('batch index: ', src.shape)
            yy = model(src,trg)
            #print('yy: \n',yy)
            loss = torch.sqrt(loss_fn(yy,trg.squeeze()))
            losses_batch_eval.append(loss.detach().numpy())
        losses_epoch_eval.append(sum(losses_batch_eval)/(batch_index+1))
        x_eval.append(epoch)
        print(f'{epoch } Val loss : {sum(losses_batch_eval)/(batch_index+1)}')

# print(yy, trg)
rmse = torch.sqrt(loss_fn(yy.data, trg.squeeze()))
print('RMSE : ', rmse)
plt.figure(figsize=(6,4))
plt.plot(losses_epoch)
plt.plot(x_eval, losses_epoch_eval)
plt.legend(['Train','Validation'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train loss (RMSE)')

# TODO plot the truth and predicted signals
y_predicted = []
y_truth = []
for batch_index, (src, trg) in enumerate(test_loader):
    model.eval()
    y_pred = model(src, trg)
    y_predicted.append(y_pred.squeeze())
    y_truth.append(trg.squeeze())
pred_signals = torch.hstack(y_predicted).unsqueeze(0)
true_signals = torch.hstack(y_truth).unsqueeze(0)
print(true_signals.shape, pred_signals.shape)

# plot the predicted and truth signals
for i in range(num_time_series):
    plt.figure()
    error = torch.sqrt(loss_fn(true_signals[i,:], pred_signals[i,:]))
    plt.plot(true_signals[i,:].detach().numpy())
    plt.plot(pred_signals[i,:].detach().numpy())
    plt.legend(['Truth','Predicted'])
    plt.title(f'Signal {i} -> RMSE: {error}') 
    plt.xlabel('time points')
    plt.ylabel('magnitude')
    plt.show()
    plt.close()

# PE = PositionalEncoding(4,4)
# print(PE(torch.randn((4,4))))

# PE1 = PositionalEncoding(8, 3)
# print(PE1(torch.randn(32,8,3)))



# P = PositionalEncoding(254, 100)
# cax = plt.matshow(P(100))
# plt.gcf().colorbar(cax)
# plt.title('The positional encoding matrix for feat_dim=512, sequence length=100')
# plt.xlabel('Features')
# plt.ylabel('position')

# import numpy as np
# import matplotlib.pyplot as plt

# def getPositionEncoding(seq_len, d, n=10000):
#     P = np.zeros((seq_len, d))
#     for k in range(seq_len):
#         for i in np.arange(int(d/2)):
#             denominator = np.power(n, 2*i/d)
#             P[k, 2*i] = np.sin(k/denominator)
#             P[k, 2*i+1] = np.cos(k/denominator)
#     return P

# P = getPositionEncoding(seq_len=4, d=4, n=100)
# print(P)

# TODO Test the model on real data

