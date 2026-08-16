import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloaderkeshi import Datases_loader as dataloader
from src.Net import Net_final

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 1
imgsz = 256

# --- CHECKPOINT SAVED DIRECTLY TO YOUR DRIVE ---
savedir = r'/content/drive/MyDrive/CMCNet_CrossGated_20260816_155857.pth' 

# Local fast directory unzipped in Step 1 of training
imgdir = r'/content/DeepCrack/test_img'
labdir = r'/content/DeepCrack/test_lab'

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

model = Net_final().to(device)
model.load_state_dict(torch.load(savedir, map_location=device), strict=False)
model.eval()

thresholds = np.arange(0.10, 0.95, 0.05)
stats = {t: {'tp': 0, 'fp': 0, 'fn': 0, 'c_inter': 0, 'c_union': 0, 'b_inter': 0, 'b_union': 0} for t in thresholds}

print(f"Evaluating {savedir} across probability thresholds...")
with torch.no_grad():
    for samples in testsets:
        img, lab = samples['image'].to(device), samples['mask'].to(device)
        logits = model(img)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        labels = (lab.cpu().numpy().squeeze() > 0.5).astype(bool)

        for t in thresholds:
            preds = probs > t
            tp = np.sum(preds & labels)
            fp = np.sum(preds & (~labels))
            fn = np.sum((~preds) & labels)
            
            stats[t]['tp'] += tp
            stats[t]['fp'] += fp
            stats[t]['fn'] += fn
            stats[t]['c_inter'] += tp
            stats[t]['c_union'] += np.sum(preds | labels)
            stats[t]['b_inter'] += np.sum((~preds) & (~labels))
            stats[t]['b_union'] += np.sum((~preds) | (~labels))

print("\n" + "="*65)
print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'mIoU':<10}")
print("="*65)

best_f1, best_t = 0.0, 0.50
for t in thresholds:
    s = stats[t]
    prec = s['tp'] / (s['tp'] + s['fp'] + 1e-7)
    rec  = s['tp'] / (s['tp'] + s['fn'] + 1e-7)
    f1   = 2 * prec * rec / (prec + rec + 1e-7)
    c_iou = s['c_inter'] / (s['c_union'] + 1e-7)
    b_iou = s['b_inter'] / (s['b_union'] + 1e-7)
    miou = (c_iou + b_iou) / 2.0

    if f1 > best_f1:
        best_f1, best_t = f1, t

    print(f"{t:<10.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {miou:<10.4f}")

print("="*65)
print(f"Optimal Threshold: {best_t:.2f} (Peak F1: {best_f1:.4f})\n")