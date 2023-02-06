import numpy as np
import matplotlib.pyplot as plt


# split samples into train and test
# def get_splits(dataset: custom_Dataset, splits):
#     n_graphs = len(dataset)
#     split_train = splits
#     i = int(np.ceil(n_graphs*split_train))
#     #print(i,j)
#     train = dataset[:i]
#     test = dataset[i:]

#     return train, test

# TODO generate a simulated dataset
# plot the signals 
def plot_signals(y_, num_time_series= 1, type = ''):
    #print('y_.shape: ',y_.shape)
    for i in range(num_time_series):
        plt.figure()
        plt.plot(y_[i,:])
        plt.legend(['signal'])
        plt.title(f'{type} signal {i}') 
        plt.show()
        plt.close()