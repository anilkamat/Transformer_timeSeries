from torch.utils.data import Dataset
import torch
import numpy as np

class custom_dataset(Dataset):
    def __init__(self, data_dir,num_hist, num_pred, data= None):
        super (custom_dataset, self).__init__()
        self.Dir = data_dir
        self.data = data #.double()
        self.num_hist = num_hist
        self.num_pred = num_pred

    def __len__(self):
        length = len(self.Data)
        return length

    def __getitem__(self, index):
        # print('index',index)
        input_data_window = self.data[index:index+self.num_hist]   # history time stamps
        output_data_window = self.data[index+self.num_hist:index+self.num_hist+self.num_pred] # future time stamps

        return input_data_window, output_data_window  # in (X,Y)  form
