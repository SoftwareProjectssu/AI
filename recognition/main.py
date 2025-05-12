import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import ThreeHundredWDataset
from model import DViT
from loss import LandmarkLoss
from utils import generate_heatmap, soft_argmax, calculate_nme
import argparse
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DViT(nstack=args.nstack).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    
    criterion = LandmarkLoss(nstack=args.nstack).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    train_dataset = ThreeHundredWDataset(args.data_dir, is_train=True)
    val_dataset = ThreeHundredWDataset(args.data_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)
    
    writer = SummaryWriter(args.log_dir)
    best_nme = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            images = batch['image'].to(device)
            landmarks = batch['landmarks'].to(device)
            
            heatmaps = generate_heatmap(landmarks, (256, 256))
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, landmarks, heatmaps)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        model.eval()
        total_nme = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                landmarks = batch['landmarks'].to(device)
                
                outputs = model(images)
                pred_landmarks = soft_argmax(outputs[-1])
                
                nme = calculate_nme(pred_landmarks, landmarks, (256, 256))
                total_nme += nme
        
        avg_nme = total_nme / len(val_loader)
        writer.add_scalar('NME/val', avg_nme, epoch)
        
        if avg_nme < best_nme:
            best_nme = avg_nme
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
        
        scheduler.step()
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nstack', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs/dvit_300w')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
