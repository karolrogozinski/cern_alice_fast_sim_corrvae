import numpy as np

import torch
from torch.utils.data import TensorDataset


data = np.load('./data/data_nonrandom_responses.npz')["arr_0"]
height, width = (data.shape[1], data.shape[2])
 
max_data = np.max(data)
max_sum = np.max([img.sum() for img in data])

data = torch.Tensor(np.float32(np.log(data+1)))
data_torch = torch.Tensor(data).unsqueeze(1)

y_data = np.ones([1, 7])

# All have to be in 0-1 range
for img in data:
    # max coords
    x_max = (img.argmax() % height) / height
    y_max = (img.argmax() // height) / height

    # sum of pixels
    weights_sum = img.sum()
    var_sum = weights_sum / max_sum

    # mass center
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height), torch.arange(width))

    x_sum = (x_coords * img).sum()
    y_sum = (y_coords * img).sum()

    x_mass = x_sum / weights_sum / 44
    y_mass = y_sum / weights_sum / 44

    # Size not 0
    not_0 = np.count_nonzero(img) / (height * width)

    # Max value
    m = (img.max()) / max_data

    y_data = np.concatenate((y_data, np.asarray(
        [[x_max, y_max, x_mass, y_mass, not_0, var_sum, m]])), axis=0)

y_data = np.float32(y_data[1:])
y_torch = torch.tensor(y_data)

dataset = TensorDataset(data_torch, y_torch)


torch.save({
    'features': dataset.tensors[0],
    'labels': dataset.tensors[1]
}, './data/dataset_nonrandom_responses.pth')
