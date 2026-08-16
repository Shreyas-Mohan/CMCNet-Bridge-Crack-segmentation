import os
import glob
from datetime import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from src.Net import Net_final
from utils.dataloaderkeshi import Datases_loader as dataloader
from utils.loss import Loss

def compute_confusion_matrix(predicted, expected):
    part = np.logical_xor(predicted, expected)
    pcount = np.bincount(part)
    tp_list = list(np.logical_and(predicted, expected))
    fp_list = list(np.logical_and(predicted, np.logical_not(expected)))
    tp = tp_list.count(1)
    fp = fp_list.count(1)
    tn = pcount[0] - tp
    fn = pcount[1] - fp
    return tp, fp, tn, fn

def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, F1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchsz = 4
lr = 0.001
items = 20  # 20 Epochs for prototyping
imgsz = 256

model = Net_final().to(device)

drive_save_dir = r'/content/drive/MyDrive'
os.makedirs(drive_save_dir, exist_ok=True)

optimizer = optim.AdamW(model.parameters(), lr=lr)

# Smooth cosine decay to match paper protocol
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=items, eta_min=1e-6)

# Assumes your working directory is where DeepCrack is unzipped
imgpath = r'DeepCrack/train_img'
labpath = r'DeepCrack/train_lab'

dataset = dataloader(imgpath, labpath, imgsz, imgsz)
trainsets = DataLoader(dataset, batch_size=batchsz, shuffle=True)
criterion = Loss()

accuracy, precision, recall, F1, ls_loss = [], [], [], [], []

def train():
    print(f"Training started. Models will be saved directly to: {drive_save_dir}")
    for epoch in range(items):
        lossx = 0
        tp, tn, fp, fn = 0, 0, 0, 0

        for idx, samples in enumerate(trainsets):
            img, lab = samples['image'], samples['mask']
            img, lab = img.to(device), lab.to(device)

            optimizer.zero_grad()
            pred = model(img)

            loss = criterion(pred, lab)
            loss.backward()

            optimizer.step()

            lossx += loss.item()

            p = pred.reshape(-1)
            p[p >= 0.] = 1
            p[p < 0.] = 0
            t = lab.reshape(-1)
            tp_, fp_, tn_, fn_ = compute_confusion_matrix(p.detach().cpu().numpy(), t.detach().cpu().numpy())
            tp += tp_
            fp += fp_
            tn += tn_
            fn += fn_

        accuracy_, precision_, recall_, F1_ = compute_indexes(tp, fp, tn, fn)
        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)
        F1.append(F1_)

        scheduler.step()
        lossx /= len(trainsets)
        ls_loss.append(lossx)

        print(f"Epoch {epoch + 1}/{items}, Loss: {lossx:.4f}, "
              f"Accuracy: {accuracy_:.4f}, Precision: {precision_:.4f}, "
              f"Recall: {recall_:.4f}, F1: {F1_:.4f}")

    # Save directly to Google Drive with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_save_path = os.path.join(drive_save_dir, f"CMCNet_CrossGated_{timestamp}.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"\n✅ Training complete! Weights permanently saved to: {final_save_path}")

    # Save logs to Drive as well
    str_result = (f"accuracy:{accuracy}\nprecision:{precision}\n"
                  f"recall:{recall}\nF1:{F1}\nloss:{ls_loss}")
    log_filename = os.path.join(drive_save_dir, f"CMCNet_log_{timestamp}.txt")
    with open(log_filename, mode='w', newline='') as f:
        f.writelines(str_result)

if __name__ == '__main__':
    train()