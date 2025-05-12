import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

def soft_argmax(heatmaps):
    B, C, H, W = heatmaps.shape
    device = heatmaps.device
    
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    y, x = torch.meshgrid(y, x)
    
    x = x.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    y = y.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    
    x_coords = (x * heatmaps).sum(dim=(2, 3))
    y_coords = (y * heatmaps).sum(dim=(2, 3))
    
    return torch.stack([x_coords, y_coords], dim=-1)

def generate_heatmap(landmarks, img_size, sigma=2):
    B, N, _ = landmarks.shape
    device = landmarks.device
    
    x = torch.linspace(-1, 1, img_size[1], device=device)
    y = torch.linspace(-1, 1, img_size[0], device=device)
    y, x = torch.meshgrid(y, x)
    
    x = x.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    y = y.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    
    landmarks_x = landmarks[..., 0].unsqueeze(-1).unsqueeze(-1)
    landmarks_y = landmarks[..., 1].unsqueeze(-1).unsqueeze(-1)
    
    heatmaps = torch.exp(-((x - landmarks_x) ** 2 + (y - landmarks_y) ** 2) / (2 * sigma ** 2))
    return heatmaps

def visualize_landmarks(image, landmarks, save_path=None):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
    
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for x, y in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    if save_path:
        cv2.imwrite(save_path, image)
    
    return image

def calculate_nme(pred_landmarks, gt_landmarks, img_size):
    if isinstance(pred_landmarks, torch.Tensor):
        pred_landmarks = pred_landmarks.cpu().numpy()
    if isinstance(gt_landmarks, torch.Tensor):
        gt_landmarks = gt_landmarks.cpu().numpy()
    
    pred_landmarks = pred_landmarks.reshape(-1, 68, 2)
    gt_landmarks = gt_landmarks.reshape(-1, 68, 2)
    
    pred_landmarks[..., 0] *= img_size[1]
    pred_landmarks[..., 1] *= img_size[0]
    gt_landmarks[..., 0] *= img_size[1]
    gt_landmarks[..., 1] *= img_size[0]
    
    left_eye = np.mean(gt_landmarks[:, 36:42], axis=1)
    right_eye = np.mean(gt_landmarks[:, 42:48], axis=1)
    interocular = np.linalg.norm(left_eye - right_eye, axis=1)
    
    nme = np.mean(np.linalg.norm(pred_landmarks - gt_landmarks, axis=2), axis=1) / interocular
    return np.mean(nme)

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)
