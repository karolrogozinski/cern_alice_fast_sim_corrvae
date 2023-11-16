import numpy as np

import torch
from torch.utils.data import TensorDataset


data = np.load('./data/data_nonrandom_responses.npz')["arr_0"]
data = torch.Tensor(np.float32(np.log(data+1)))

data_torch = torch.Tensor(data).unsqueeze(1)

y_data = np.ones([1, 2])

for img in data:
    # TODO Normalise size
    x = (img.argmax() % 44) / 44
    y = (img.argmax() // 44) / 44
    # size = np.count_nonzero(img)
    # m = (img.max())
    y_data = np.concatenate((y_data, np.asarray([[x, y]])), axis=0)

y_data = np.float32(y_data[1:])
y_torch = torch.tensor(y_data)

dataset = TensorDataset(data_torch, y_torch)


torch.save({
    'features': dataset.tensors[0],
    'labels': dataset.tensors[1]
}, './data/dataset_nonrandom_responses.pth')
