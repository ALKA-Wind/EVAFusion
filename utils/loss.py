"""Fusion losses extracted from original DCEvo utils/loss.py (no detection imports)."""
import torch
from torch import nn
from torch.nn import functional as F


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weightx', torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('weighty', torch.tensor(
            [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3))

    def forward(self, x):
        return (F.conv2d(x, self.weightx, padding=1).abs()
                + F.conv2d(x, self.weighty, padding=1).abs())


class Fusionloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1]
        intensity = F.l1_loss(torch.maximum(image_y, image_ir), generate_img)
        gradient = F.l1_loss(torch.maximum(self.sobelconv(image_y), self.sobelconv(image_ir)),
                             self.sobelconv(generate_img))
        return intensity + 10 * gradient, intensity, gradient


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    img1, img2 = img1.flatten(2), img2.flatten(2)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    numerator = (img1 * img2).sum(dim=-1)
    denominator = eps + img1.square().sum(dim=-1).sqrt() * img2.square().sum(dim=-1).sqrt()
    return (numerator / denominator).clamp(-1, 1).mean()
