"""Utils for visualizations in notebooks"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def show_all_keypoints(image, keypoints, pred_kpts=None):
    """Show image with predicted keypoints"""
    height, width = image.shape[1:]
    image = (image.clone() * 255).view(height, width)
    plt.imshow(image, cmap='gray')
    if pred_kpts is not None:
        loss = F.mse_loss(pred_kpts, keypoints)
        plt.title('Loss: %s' % str(loss))
    half_w_h = torch.tensor([width, height], dtype=keypoints.dtype, device=keypoints.device) / 2
    keypoints = keypoints.clone() * half_w_h + half_w_h
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=200, marker='.', c='m')
    if pred_kpts is not None:
        pred_kpts = pred_kpts.clone() * half_w_h + half_w_h
        plt.scatter(pred_kpts[:, 0], pred_kpts[:, 1], s=200, marker='.', c='r')
    plt.show()
