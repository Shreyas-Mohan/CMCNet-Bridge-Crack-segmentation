import os
import glob
import argparse
from datetime import datetime

# Prevent CUDA memory fragmentation on 16GB GPUs
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from src.Net import Net_final
from utils.dataloaderkeshi import Datases_loader as dataloader
from utils.loss import Loss

def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, F1

def parse_args():
    parser = argparse.ArgumentParser(description="CMC-Net Training for Crack Segmentation")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=512, help="Native image resolution (default: 512)")
    parser.add_argument("--batchsz", type=int, default=2, help="Batch size per forward pass (default: 2, saves VRAM)")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps (default: 2, effective batch size = 4)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable Automatic Mixed Precision (default: False for pytorch_wavelets stability)")
    parser.add_argument("--data_dir", type=str, default="DeepCrack", help="Path to DeepCrack dataset directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints and logs")
    return parser.parse_args()

def train():
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if args.save_dir:
        save_dir = args.save_dir
    else:
        # Save directly in the project folder alongside existing .pth models
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints and logs will be saved to: {save_dir}")

    # Initialize model
    model = Net_final().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = Loss()

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"AMP (Mixed Precision): {'Enabled' if use_amp else 'Disabled'}")

    # Auto-detect data_dir if specified directory is not found
    data_dir = args.data_dir
    if not os.path.exists(os.path.join(data_dir, 'train_img')):
        candidates = [
            '/content/DeepCrack/DeepCrack',
            '/content/DeepCrack',
            '/content/DeepCrack_Dataset/DeepCrack',
            '/content/CMCNet-main-btp/DeepCrack',
            'DeepCrack',
            os.path.join(os.getcwd(), 'DeepCrack')
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, 'train_img')):
                data_dir = cand
                break

    imgpath = os.path.join(data_dir, 'train_img')
    labpath = os.path.join(data_dir, 'train_lab')

    if not os.path.exists(imgpath):
        raise FileNotFoundError(
            f"Could not find 'train_img' in '{data_dir}'. "
            f"Please ensure DeepCrack is unzipped or pass --data_dir /path/to/DeepCrack"
        )

    print(f"Using dataset from: {data_dir}")
    dataset = dataloader(imgpath, labpath, args.imgsz, args.imgsz)
    trainsets = DataLoader(
        dataset, 
        batch_size=args.batchsz, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Dataset: {len(dataset)} samples | {len(trainsets)} batches per epoch at {args.imgsz}x{args.imgsz}")
    print(f"Training for {args.epochs} epochs with batch size {args.batchsz} (accum_steps={args.accum_steps})...\n")

    accuracy, precision, recall, F1, ls_loss = [], [], [], [], []
    best_f1 = 0.0
    best_epoch = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(args.epochs):
        model.train()
        lossx = 0.0
        tp, tn, fp, fn = 0, 0, 0, 0
        optimizer.zero_grad()

        for idx, samples in enumerate(trainsets):
            img, lab = samples['image'].to(device, non_blocking=True), samples['mask'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(img)
                loss = criterion(pred, lab)
                if args.accum_steps > 1:
                    loss = loss / args.accum_steps

            if use_amp:
                scaler.scale(loss).backward()
                if (idx + 1) % args.accum_steps == 0 or (idx + 1) == len(trainsets):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (idx + 1) % args.accum_steps == 0 or (idx + 1) == len(trainsets):
                    optimizer.step()
                    optimizer.zero_grad()

            loss_val = loss.item() * (args.accum_steps if args.accum_steps > 1 else 1)
            lossx += loss_val

            # Blazing fast confusion matrix computation on GPU directly
            with torch.no_grad():
                p = pred > 0.0
                t = lab > 0.5
                tp += (p & t).sum().item()
                fp += (p & (~t)).sum().item()
                fn += ((~p) & t).sum().item()
                tn += ((~p) & (~t)).sum().item()

        accuracy_, precision_, recall_, F1_ = compute_indexes(tp, fp, tn, fn)
        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)
        F1.append(F1_)

        scheduler.step()
        lossx /= len(trainsets)
        ls_loss.append(lossx)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1:3d}/{args.epochs}] Loss: {lossx:.4f} | "
              f"Acc: {accuracy_:.4f} | Prec: {precision_:.4f} | "
              f"Rec: {recall_:.4f} | F1: {F1_:.4f} | LR: {current_lr:.6f}")

        # Save best model checkpoint
        if F1_ > best_f1:
            best_f1 = F1_
            best_epoch = epoch + 1
            best_save_path = os.path.join(save_dir, f"CMCNet_best_f1.pth")
            torch.save(model.state_dict(), best_save_path)

        # Periodic checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            periodic_path = os.path.join(save_dir, f"CMCNet_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), periodic_path)

    # Save final model
    final_save_path = os.path.join(save_dir, f"CMCNet_CrossGated_512x512_{timestamp}_final.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"\n✅ Training complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Saved best weights to: {os.path.join(save_dir, 'CMCNet_best_f1.pth')}")
    print(f"Saved final weights to: {final_save_path}")

    # Save training logs
    str_result = (f"epochs:{args.epochs}\nimgsz:{args.imgsz}\nbatchsz:{args.batchsz}\n"
                  f"best_f1:{best_f1}\nbest_epoch:{best_epoch}\n"
                  f"accuracy:{accuracy}\nprecision:{precision}\n"
                  f"recall:{recall}\nF1:{F1}\nloss:{ls_loss}\n")
    log_filename = os.path.join(save_dir, f"CMCNet_log_512x512_{timestamp}.txt")
    with open(log_filename, mode='w', newline='') as f:
        f.writelines(str_result)
    print(f"Logs saved to: {log_filename}")

if __name__ == '__main__':
    train()