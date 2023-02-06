import numpy as np
import torch

def split_timeseries(v, num_time_series):
    length = int(len(v)/num_time_series)
    V =[]
    for i in range(0,length*num_time_series,length):
        V.append(v[i:i+length])
    V = np.vstack(V).T
    return V

def min_max_normalization(v): 
#    print(v.shape)
    v = v.unsqueeze(1).T
    # print(v.shape)
    m,_ = v.shape  #m->rows, n->columns.
    # m = len(v)

    mins = []
    maxs = []
    for i in range(int(m)):
        mins.append(v[i,:].min())
        maxs.append(v[i,:].max())
        v[i,:] =  (v[i,:]-v[i,:].min())/(v[i,:].max()-v[i,:].min())
    return v, mins, maxs  
    
# check RMSE 
def RMSE(v, v_):
    # print(v.shape, v_.shape)
    # print('rmse ....',torch.sqrt(torch.mean((torch.squeeze(v)-torch.squeeze(v_))**2, axis=0)))
    return torch.mean(torch.sqrt(torch.mean((torch.squeeze(v)-torch.squeeze(v_))**2, axis=0)))

def noise(v,v_):
    """ returns difference between two vectors"""
    return v-v_

def norm_to_real_domain(v, y_min, y_max):
    m,_  = v.shape
    print('v.shape: ',v.shape, y_min, y_max)
    for i in range(m): 
        v[i,:] = v[i,:]* (y_max[i]-y_min[i])+y_min[i]
    return v
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
                self.early_stop = True
        return self.early_stop

    