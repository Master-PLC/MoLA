import ot
import torch


def cal_wasserstein(X1, X2, distance, ot_type='sinkhorn', normalize=1, numItermax=10000, stopThr=1e-4, reg_sk=0.1, reg_m=10):
    """
    X1: prediction sequence with shape [B, T, D]
    X2: label sequence with shape [B, T, D]
    distance: the method to calculate the pair-wise sample distance
    ot_type: the definition of OT problem
    reg_sk: the strength of entropy regularization in Sinkhorn
    reg_m: the strength of mass-preservation constraint in UOT
    Currently we can fix the ot_type as sinkhorn, and there are key parameters to tune:
        1. whether normalize is needed?
        2. which distance is better? time/fft_2norm/fft_1norm, other distances do not need to investigate now.
        3. the optimum reg_sk? Possible range: [0.01-5]
    Moreover, the weight of the loss should be tuned in this duration. Try both relative weights (\alpha for wass, 1-\alpha for mse) and absolute weights (\alpha for wass, 1 for mse)
    """
    if distance == 'time':
        M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
    elif distance == 'fft_2norm':
        X1 = torch.fft.rfft(X1.transpose(1, 2))
        X2 = torch.fft.rfft(X2.transpose(1, 2))
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()**2).sum(-1)
    elif distance == 'fft_1norm':
        X1 = torch.fft.rfft(X1.transpose(1, 2))
        X2 = torch.fft.rfft(X2.transpose(1, 2))
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()).sum(-1)
    elif distance == 'fft_mag':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
        X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
        M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
    elif distance == 'fft_mag_abs':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
        X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
        M = torch.norm(X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:], p=1, dim=2)
    elif distance == 'fft_2norm_kernel':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)
        sigma = 1
        M = -1 * torch.exp(-1 * ((X1[:,None,:] - X2[None,:,:]).abs()).sum(-1) / 2/sigma)
    elif distance == 'fft_2norm_multikernel':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)
        w = [0.5, 0.5]
        sigma = [0.1, 1]
        dist_weighted = [-1 * w[i] * torch.exp(-1 * ((X1[:,None,:] - X2[None,:,:]).abs()).sum(-1) / 2/sigma[i]) for i in range(len(w))]
        M = sum(dist_weighted)

    if normalize == 1:
        M = M / M.max()

    a, b = torch.ones((len(X1),), device=M.device) / len(X1), torch.ones((len(X1),), device=M.device) / len(X1)

    if ot_type == 'sinkhorn':
        pi = ot.sinkhorn(a, b, M, reg=reg_sk, max_iter=numItermax, tol_rel=stopThr).detach()
    elif ot_type == 'emd':
        pi = ot.emd2(a, b, M, numItermax=numItermax).detach()
    elif ot_type == 'uot':
        pi = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=reg_sk, stopThr=stopThr, numItermax=numItermax, reg_m=reg_m).detach()
    elif ot_type == 'uot_mm':
        pi = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, c=None, reg=0, div='kl', G0=None, numItermax=numItermax, stopThr=stopThr).detach()

    loss = (pi * M).sum()
    return loss