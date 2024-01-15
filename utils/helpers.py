import torch

import matplotlib.pyplot as plt


def plot_epoch(data_loader, model, device, taus, epoch, time):
    print("===============================================================")
    print("Completed Epoch", epoch, " Time: ", time)
    print("===============================================================")

    for (data, label) in data_loader:
        data = data.to(device)
        cond = label[:, 9:].to(device)

        (reconstruct, _, _), _, _, _, latent_sample_z, latent_sample_w, \
            latent_sample_u, _, _ = model(data, cond, taus)

        fig, axs = plt.subplots(4, 7, figsize=(15, 6))
        for i in range(28):
            if i < 7:
                x = data.cpu().detach().numpy()[i][0]
            elif i < 14:
                x = reconstruct.cpu().detach().numpy()[i-7][0]
            elif i < 21:
                (reconstruct_new_w, _, _), _, _, _, _, _, _, _, _ = model(
                    data[-1], cond[-1].unsqueeze(0), taus,
                    w2=latent_sample_w[i-14],
                    c2=latent_sample_u[i-14])
                x = reconstruct_new_w.cpu().detach().numpy()[0][0]
            else:
                (reconstruct_new_w, _, _), _, _, _, _, _, _, _, _ = model(
                    data[-1], cond[-1].unsqueeze(0), taus,
                    w2=latent_sample_w[i-21])
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
