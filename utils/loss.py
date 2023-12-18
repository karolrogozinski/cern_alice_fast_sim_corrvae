import math

import torch


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


def _kl_normal_loss(mean, logvar):
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    return total_kl


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist):
    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


def _get_log_pzw_qzw_prodzw_qzwCx(
            latent_sample_z, latent_sample_w, latent_dist_z,
            latent_dist_w, n_data, is_mss=True
        ):
    batch_size, _ = latent_sample_z.shape
    batch_size, _ = latent_sample_w.shape

    latent_dist = (torch.cat([latent_dist_z[0], latent_dist_w[0]], dim=-1),
                   torch.cat([latent_dist_z[1], latent_dist_w[1]], dim=-1))

    latent_sample = torch.cat([latent_sample_z, latent_sample_w], dim=-1)

    # calculate log q(z,w|x)
    log_q_zwCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z,w)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pzw = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qzqw = matrix_log_density_gaussian(latent_sample, *latent_dist)
    mat_log_qz = matrix_log_density_gaussian(latent_sample_z, *latent_dist_z)
    mat_log_qw = matrix_log_density_gaussian(latent_sample_w, *latent_dist_w)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(
            batch_size, n_data).to(latent_sample.device)
        mat_log_qzqw = mat_log_qzqw + log_iw_mat.view(
            batch_size, batch_size, 1)
        log_iw_mat_z = log_importance_weight_matrix(
            batch_size, n_data).to(latent_sample_z.device)
        mat_log_qz = mat_log_qz + log_iw_mat_z.view(
            batch_size, batch_size, 1)
        log_iw_mat_w = log_importance_weight_matrix(
            batch_size, n_data).to(latent_sample_w.device)
        mat_log_qw = mat_log_qw + log_iw_mat_w.view(
            batch_size, batch_size, 1)

    log_qzw = torch.logsumexp(mat_log_qzqw.sum(2), dim=1, keepdim=False)
    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_qw = torch.logsumexp(mat_log_qw.sum(2), dim=1, keepdim=False)
    log_prod_qzqw = log_qz + log_qw

    return log_pzw, log_qzw, log_prod_qzqw, log_q_zwCx


def reg_mask(mask):
    l1norm = torch.sum(torch.abs(mask))

    return l1norm


def get_losses(latent_dist, latent_sample_w, latent_dist_w, beta,
               latent_sample_z, latent_dist_z, w_mask, device, idx_kl,
               rec_loss, rec_loss_prop_all, w_kl, loader_size, lambdas,
               latent_sample_cond, latent_dist_cond):

    kl_loss = _kl_normal_loss(*latent_dist)

    _, log_qw, log_prod_qwi, _ = _get_log_pz_qz_prodzi_qzCx(latent_sample_w,
                                                            latent_dist_w)

    tc_loss = (log_qw - log_prod_qwi).mean()
    pairwise_tc_loss = beta * tc_loss

    _, log_qwz, log_prod_qwqz, _ = _get_log_pzw_qzw_prodzw_qzwCx(
        latent_sample_z, latent_sample_w, latent_dist_z,
        latent_dist_w, loader_size
    )
    # groupwise_tc_loss = beta * (log_qwz - log_prod_qwqz).mean()
    groupwise_tc_loss = 0

    _, log_qwc, log_prod_qwqc, _ = _get_log_pzw_qzw_prodzw_qzwCx(
        latent_sample_cond, latent_sample_w, latent_dist_cond,
        latent_dist_w, loader_size
    )
    groupwise_tc_loss_cond = beta * (log_qwc - log_prod_qwqc).mean()
    # groupwise_tc_loss_cond = 0

    l1norm = reg_mask(w_mask).to(device)

    if idx_kl <= 100000:
        loss = lambdas[0]*rec_loss + pairwise_tc_loss + lambdas[2]*rec_loss_prop_all\
            + groupwise_tc_loss + groupwise_tc_loss_cond + 100000*l1norm
    else:
        if w_kl < 100000:
            w_kl += 1
        loss = lambdas[0]*rec_loss + lambdas[1]*pairwise_tc_loss +\
            lambdas[2]*rec_loss_prop_all + lambdas[3]*groupwise_tc_loss +\
            lambdas[4]*groupwise_tc_loss_cond + lambdas[5] * w_kl * kl_loss

    return kl_loss, pairwise_tc_loss, groupwise_tc_loss, \
        groupwise_tc_loss_cond, l1norm, loss, w_kl
