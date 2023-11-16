import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


def prepare_dataloader(data, batch_size):
    data = np.float32(np.log(data+1))

    max_data = np.max(data)

    data = data / max_data

    transform = transforms.Pad((10, 10))

    data_torch = torch.Tensor(data)
    data_torch = transform(data_torch).unsqueeze(1)

    y_data = np.ones([1, 2])

    for img in data:
        # TODO Normalise size

        x = (img.argmax() % 64) / 64
        y = (img.argmax() // 64) / 64
        # size = np.count_nonzero(img)
        # m = (img.max())
        y_data = np.concatenate((y_data, np.asarray([[x, y]])), axis=0)

    y_data = y_data[1:]
    y_data = np.float32(y_data)

    y_torch = torch.tensor(y_data)

    dataset = TensorDataset(data_torch, y_torch)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataloader


def plot_epoch(data_loader, model, device, taus, epoch, time):
    print("===============================================================")
    print("Completed Epoch", epoch, " Time: ", time)
    print("===============================================================")

    for data, _ in data_loader:
        data = data.to(device)
        (reconstruct, _), _, _, _, latent_sample_w, _, _ = model(data, taus)

        fig, axs = plt.subplots(3, 7, figsize=(15, 6))
        for i in range(21):
            if i < 7:
                x = data.cpu().detach().numpy()[i][0]
            elif i < 14:
                x = reconstruct.cpu().detach().numpy()[i-7][0]
            else:
                (reconstruct_new_w, _), _, _, _, _, _, _ = model(
                    data[-1], taus, w2=latent_sample_w[i-14])
                x = reconstruct_new_w.cpu().detach().numpy()[0][0]

            im = axs[i//7, i % 7].imshow(x, interpolation='none',
                                         cmap='gnuplot')
            axs[i//7, i % 7].axis('off')
            fig.colorbar(im, ax=axs[i//7, i % 7])
        plt.show()
        break


def save_model(model, optimizer, results_dir, epoch):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{results_dir}modelCorrVAE_{epoch}.pt')
