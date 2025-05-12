import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import generate_heatmap, soft_argmax

class AwingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

        A = omega * math.log(1 + (theta / epsilon) ** alpha)
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))

    def forward(self, pred, target):
        delta = (pred - target).abs()
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + (delta / self.epsilon) ** self.alpha),
            self.omega / self.alpha * (delta - self.theta) + self.A
        )
        return loss.mean()

class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        delta = (target - pred).abs()
        delta_abs = delta

        mask = (delta_abs < self.omega).float()
        mask2 = (delta_abs >= self.omega).float()

        y = mask * self.omega * torch.log(1 + delta_abs / self.epsilon) + mask2 * (delta_abs - self.omega / 2)

        return y.mean()

class LandmarkLoss(nn.Module):
    def __init__(self, nstack=8, beta=0.5):
        super(LandmarkLoss, self).__init__()
        self.nstack = nstack
        self.wing_loss = WingLoss()
        self.awing_loss = AwingLoss()
        self.beta = beta 

    def forward(self, outputs, target_coords, target_heatmaps=None):
        weights = [1.2**(i - len(outputs)) for i in range(len(outputs))]
        
        total_loss = 0
        for i, (w, out) in enumerate(zip(weights, outputs)):
            loss = 0

            if target_heatmaps is not None:
                heatmap_loss = self.awing_loss(out, target_heatmaps)
                loss += heatmap_loss

            pred_coords = soft_argmax(out)
            coord_loss = self.wing_loss(pred_coords, target_coords)
            loss += coord_loss

            total_loss += w * loss        
        return total_loss / self.nstack

class DViTLoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.coord_loss = nn.SmoothL1Loss()
        self.heatmap_loss = nn.MSELoss()
        self.beta = beta

    def forward(self, outputs, target_heatmaps, target_coords):
        loss = 0
        for out in outputs:
            pred_coords = soft_argmax(out)
            loss += self.coord_loss(pred_coords, target_coords) + self.beta * self.heatmap_loss(out, target_heatmaps)
        return loss / len(outputs)

# def soft_argmax(heatmap):
#     pass

def calculate_nme(pred_landmarks, gt_landmarks, image_size):
    H, W = image_size
    scale = torch.tensor([W, H], device=pred_landmarks.device)

    pred_landmarks = pred_landmarks * scale
    gt_landmarks = gt_landmarks * scale

    left_eye = gt_landmarks[:, 36]
    right_eye = gt_landmarks[:, 45]
    interocular = torch.norm(left_eye - right_eye, dim=1)

    error = torch.norm(pred_landmarks - gt_landmarks, dim=2).mean(dim=1)
    nme = (error / interocular).mean()

    return nme.item()