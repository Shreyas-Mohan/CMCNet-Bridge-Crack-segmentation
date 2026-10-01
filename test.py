import os
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloaderkeshi import Datases_loader as dataloader
from src.Net import Net_final

def parse_args():
    parser = argparse.ArgumentParser(description="CMC-Net Evaluation across Probability Thresholds")
    parser.add_argument("--imgsz", type=int, default=512, help="Image resolution for testing (default: 512)")
    parser.add_argument("--batchsz", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint file")
    parser.add_argument("--data_dir", type=str, default="DeepCrack", help="Path to DeepCrack dataset directory")
    return parser.parse_args()

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = args.batchsz
imgsz = args.imgsz

# Automatically locate checkpoint if not explicitly provided
if args.checkpoint and os.path.exists(args.checkpoint):
    savedir = args.checkpoint
elif os.path.exists(os.path.join(os.getcwd(), 'CMCNet_best_f1.pth')):
    savedir = os.path.join(os.getcwd(), 'CMCNet_best_f1.pth')
elif os.path.exists('CMCNet_best_f1.pth'):
    savedir = 'CMCNet_best_f1.pth'
elif os.path.exists('CMCNet_CrossGated_20260816_155857.pth'):
    savedir = 'CMCNet_CrossGated_20260816_155857.pth'
else:
    savedir = 'CMCNet_best_f1.pth'

# Auto-detect data_dir if specified directory is not found
data_dir = args.data_dir
if not os.path.exists(os.path.join(data_dir, 'test_img')):
    candidates = [
        '/content/DeepCrack/DeepCrack',
        '/content/DeepCrack',
        '/content/DeepCrack_Dataset/DeepCrack',
        '/content/CMCNet-main-btp/DeepCrack',
        'DeepCrack',
        os.path.join(os.getcwd(), 'DeepCrack')
    ]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, 'test_img')):
            data_dir = cand
            break

imgdir = os.path.join(data_dir, 'test_img')
labdir = os.path.join(data_dir, 'test_lab')

if not os.path.exists(imgdir):
    raise FileNotFoundError(
        f"Could not find 'test_img' in '{data_dir}'. "
        f"Please ensure DeepCrack is unzipped or pass --data_dir /path/to/DeepCrack"
    )

print(f"Using test dataset from: {data_dir}")

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

model = Net_final().to(device)
if os.path.exists(savedir):
    print(f"Loading weights from: {savedir}")
    model.load_state_dict(torch.load(savedir, map_location=device), strict=False)
else:
    print(f"Warning: Checkpoint '{savedir}' not found. Please provide valid --checkpoint path.")
model.eval()

# Extended sweep: includes fine-grained low thresholds (0.02-0.08) to capture faint hairline cracks
thresholds = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
stats = {t: {'tp': 0, 'fp': 0, 'fn': 0, 'c_inter': 0, 'c_union': 0, 'b_inter': 0, 'b_union': 0} for t in thresholds}

print(f"Evaluating {savedir} across probability thresholds ({len(testsets)} test images)...")
t0 = time.time()
with torch.no_grad():
    for idx, samples in enumerate(testsets):
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

        elapsed = time.time() - t0
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (len(testsets) - (idx + 1))
        pct = ((idx + 1) / len(testsets)) * 100

        # Detailed verbose progress every 5 images and at the final image
        if (idx + 1) % 5 == 0 or (idx + 1) == len(testsets):
            print(f"Progress: [{idx + 1:3d}/{len(testsets)}] ({pct:5.1f}%) | "
                  f"Speed: {avg_time:.2f}s/img | ETA: {int(remaining // 60)}m {int(remaining % 60):02d}s", flush=True)

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