"""Validation metrics of the scene-leakage decoder (Table 4). Inputs are float tensors in [-1, 1], shape (B, 3, H, W);
every function returns one value per image.

LPIPS: lpips 0.1.4, AlexNet backbone, linear weights v0.1 (the network that is also used in the training loss).
SSIM: Gaussian window 11x11, sigma 1.5, data range 2, averaged over pixels and channels.
"""
import torch
import torch.nn.functional as F

_lpips_net = None


def get_lpips(device):
    # The metric network is a second LPIPS instance, built on the first call (during the first validation). Building it
    # draws from the CPU random generator, which advances the data order of later epochs; keep it where it is.
    global _lpips_net
    if _lpips_net is None:
        import lpips
        _lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device)
        for p in _lpips_net.parameters():
            p.requires_grad_(False)
    return _lpips_net


@torch.no_grad()
def lpips_score(pred, target, device=None):
    device = device or pred.device
    net = get_lpips(device)
    return net(pred.to(device), target.to(device)).flatten()


def _gaussian_window(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w_2d = g[:, None] * g[None, :]
    return w_2d.expand(channels, 1, window_size, window_size).contiguous()


@torch.no_grad()
def ssim_score(pred, target, data_range=2.0):
    pred = pred.float()
    target = target.float()
    C = pred.size(1)
    win = _gaussian_window(channels=C, device=pred.device)
    mu_x = F.conv2d(pred, win, padding=5, groups=C)
    mu_y = F.conv2d(target, win, padding=5, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sx2 = F.conv2d(pred * pred, win, padding=5, groups=C) - mu_x2
    sy2 = F.conv2d(target * target, win, padding=5, groups=C) - mu_y2
    sxy = F.conv2d(pred * target, win, padding=5, groups=C) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sxy + c2)) / \
               ((mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2))
    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def psnr_score(pred, target, data_range=2.0):
    mse = ((pred.float() - target.float()) ** 2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=1e-12))
