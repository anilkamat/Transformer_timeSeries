import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils.utils import *
from Utils.custom_dataset import custom_dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


#TODO load the dataset  
def dataloader(y_,n_hist, n_pred, batch_size, split_size=0.8, shuffle = False):
    # compile the dataset into input (x) and output(y) 
    random_seed = 1024
    data_dir = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\Models\GAT_LSTM_\data'
    window_size = n_hist+n_pred
    samples = custom_dataset(data_dir,n_hist, n_pred, data= y_)
    n_samples = len(y_)-window_size+1 # formula to limit traversing on the timeseries
    # print('len of y ', len(y_))
    # print('total samples:', n_samples)
    # print('samples, y_: ',samples[10][0], y_[10:10+n_hist])
    assert (torch.eq(y_[10:10+n_hist,:].squeeze(),samples[10][0].squeeze())).all(), 'data mismatch after custom_data'
    # assert (torch.eq(torch.from_numpy(y_[n_hist:n_hist+n_pred,:]),samples[0][1])).all() , 'data mismatch after custom_data'

    indices = list(range(n_samples))
    train_split = int(split_size*n_samples)
    if shuffle: 
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices = indices[:train_split]
    test_indices = indices[train_split:]
    # print(f'trian indices {train_indices}, test indices :{test_indices}')
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_samples = [ samples[idx] for idx in train_indices]
    test_samples = [ samples[idx] for idx in test_indices]
    # print('train samples length: ',len(train_samples), train_samples[10][0])

    # train_loader = DataLoader(samples, batch_size=batch_size, shuffle= False )
    # test_loader = DataLoader(samples, batch_size=batch_size, shuffle= False)

    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle= False )
    test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle= False)
    return train_loader, test_loader

# plot the signals 
def plot_signals(y_,num_time_series=1):
    for i in range(num_time_series):
        plt.figure()
        if num_time_series ==1: 
            plt.plot(y_)
        else: 
            plt.plot(y_[:,i])
        plt.legend(['signal'])
        plt.title(f'signal {i}') 
        plt.show()
        plt.close()

# ## generate random data for testing    
# x = torch.linspace(0,2,2000)
# y = 2*torch.sin(10*torch.pi*x) #+torch.randn(len(x))
# plot_signals(y)
# num_time_series = 2
# y = split_timeseries(y, num_time_series)
# plot_signals(y,num_time_series)
# y = min_max_normalization(y) # normalize y
# plot_signals(y,num_time_series)

# n_hist = 10
# n_pred = 1
# # test the loader 
# train_data, test_data = dataloader(y,n_hist, n_pred, batch_size=1, split_size=0.8, shuffle = False)

# # plot the data from loader to check if the signals are still temporally-entact . 
# signal_ = []
# for batch_index, (input, output) in enumerate(test_data):
#     signal_.append(output.detach().numpy())
# true_signals = np.squeeze(np.vstack(signal_))
# print(true_signals.shape)
# plot_signals(true_signals,num_time_series=2)


